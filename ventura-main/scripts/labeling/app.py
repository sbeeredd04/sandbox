#!/usr/bin/env python3
"""
Flask viewer + annotation tool
• Raw RGB sent; browser composites path-mask + per-spline SAM-2 masks
• Each spline & its mask share the same colour
• Keeps an up-to-date human_eval_labels.txt (ride_name, start_frame, end_frame)
"""

from __future__ import annotations
import argparse, base64, cv2, hickle as hkl, numpy as np, torch, yaml, re, pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from scripts.inference.build_dataset import build_dataset
from scripts.labeling.sam2_helpers import (
    create_sam2, 
    compute_sam2_mask
)

# ────────── CLI ───────────────────────────────────────────────────
pa = argparse.ArgumentParser()
pa.add_argument("--dataset_cfg", default="config/dataset/planning/frodo8k_turn_goal.yaml")
pa.add_argument("--split_dir",  required=True)
pa.add_argument("--out_dir",    required=True)
pa.add_argument("--host",       default="0.0.0.0")
pa.add_argument("--port",       type=int, default=5000)
ARGS = pa.parse_args()

# ────────── dataset / globals ─────────────────────────────────────
DATASET = build_dataset(
    yaml.safe_load(Path(ARGS.dataset_cfg).read_text()),
    split_dir=Path(ARGS.split_dir).name
)
OUT_DIR = Path(DATASET.split_dir).parent / ARGS.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_CSV = OUT_DIR / "human_eval_labels.txt"       # <- new
_RX = re.compile(
    r"output_rides_(\d+)/ride_([^/_]+)_([^/_]+)_([^/_]+)/seq_(\d+)/gt_path_info\.h5$"
)

DS_LEN   = len(DATASET)
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
ANN: dict[int, dict] = {}                     # idx → {"paths": …}

# ────────── label-file utilities ──────────────────────────────────
def rebuild_labels() -> None:
    """
    Glob OUT_DIR for all gt_path_info.h5 files and rewrite LABELS_CSV,
    sorted by ride_name then start_frame.
    """
    entries: list[tuple[str, str, str]] = []
    for fp in OUT_DIR.rglob("gt_path_info.h5"):
        if (m := _RX.search(fp.as_posix())):
            rid, d0, d1, ts, sf = m.groups()
            ride_name = f"{rid} {d0} {d1} {ts}"
            entries.append((ride_name, sf, "-1"))
    df = (pd.DataFrame(entries, columns=["ride_name", "start_frame", "end_frame"])
            .sort_values(["ride_name", "start_frame"], kind="mergesort"))
    df.to_csv(LABELS_CSV, index=False)

# run once on start-up
rebuild_labels()

# ────────── colour helpers ────────────────────────────────────────
def hsl_to_rgb(h,s,l):
    c=(1-abs(2*l-1))*s; x=c*(1-abs((h/60)%2-1)); m=l-c/2
    r,g,b=[(c,x,0),(x,c,0),(0,c,x),(0,x,c),(x,0,c),(c,0,x)][int(h//60)]
    return int((b+m)*255),int((g+m)*255),int((r+m)*255)  # BGR

def colour_for_pid(pid:str)->tuple[int,int,int]:
    return hsl_to_rgb((int(pid)*67)%360,1,0.5)

# ────────── SAM-2 initialisation  ─────────────────────────────────
SAM2 = create_sam2(device=DEVICE)

# ────────── misc helpers ──────────────────────────────────────────
def tens_to_rgb(t: torch.Tensor)->np.ndarray:
    arr=t.squeeze(0).permute(1,2,0).cpu().numpy()
    return cv2.normalize(arr,None,0,255,cv2.NORM_MINMAX).astype("uint8")

png_bytes=lambda im:base64.b64encode(
    cv2.imencode(".png",cv2.cvtColor(im,cv2.COLOR_RGB2BGR))[1]).decode()

def mask_to_png(mask:np.ndarray,bgr,alpha=120)->str:
    h,w=mask.shape; img=np.zeros((h,w,4),np.uint8)
    img[mask]=[*bgr,alpha]
    return base64.b64encode(cv2.imencode(".png",img)[1]).decode()

# ────────── robust spline fit  ────────────────────────────────────
def spline_fit(raw:list[list[float]], N:int=200)->np.ndarray:
    pts=np.asarray(raw,float)
    if pts.shape[0] < 2 or np.allclose(pts, pts[0]):
        return np.repeat(pts[:1], N, 0)

    seg=np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total=seg.sum()
    if total == 0:
        return np.repeat(pts[:1], N, 0)

    s=np.concatenate(([0.0], np.cumsum(seg))) / total
    deg=min(3, pts.shape[0]-1)
    try:
        cx=np.polyfit(s, pts[:,0], deg)
        cy=np.polyfit(s, pts[:,1], deg)
    except (np.linalg.LinAlgError, TypeError):
        return np.linspace(pts[0], pts[-1], N)
    u=np.linspace(0,1,N)
    return np.vstack([np.polyval(cx,u), np.polyval(cy,u)]).T

# ────────── annotation I/O  ───────────────────────────────────────
def ann_fp(seq:str,frame:int)->Path:
    rid,d0,d1,ts=seq.split(" ")
    return OUT_DIR/f"output_rides_{rid}"/f"ride_{d0}_{d1}_{ts}"/f"seq_{frame}"/"gt_path_info.h5"

def ensure_dir(p:Path): p.mkdir(parents=True,exist_ok=True)
def prune_dirs(p:Path):
    for d in [p]+list(p.parents):
        if d==OUT_DIR or not d.exists(): break
        try:d.rmdir()
        except OSError: break

def load_ann(fp:Path):
    if not fp.exists(): return {"paths":{}}
    d=hkl.load(fp); spl=d["paths_2d"]
    goals=d.get("goals",[""]*len(spl))
    masks=d.get("goal_masks",[None]*len(spl))
    ptsls=d.get("clicked_pts",[[]for _ in spl])
    return {"paths":{str(i):{"spline":spl[i],"goal":goals[i],
                             "mask":(masks[i].astype(bool) if i<len(masks) and masks[i] is not None else None),
                             "pts":ptsls[i] if i<len(ptsls) else []}
                     for i in range(len(spl))}}

def save_ann(fp:Path,ann:dict):
    ensure_dir(fp.parent)
    p=ann["paths"]
    if not p:
        fp.unlink(missing_ok=True); prune_dirs(fp.parent); rebuild_labels(); return
    spl=np.stack([v["spline"] for v in p.values()])
    ref_mask = next((v["mask"] for v in p.values() if v["mask"] is not None), None)
    h,w = ref_mask.shape if ref_mask is not None else (256,256)
    masks=[(v["mask"].astype("bool") if v["mask"] is not None else np.zeros((h,w),bool))
           for v in p.values()]
    hkl.dump({"paths_2d":spl,
              "goals":[v["goal"] for v in p.values()],
              "goal_masks":masks,
              "clicked_pts":[v["pts"] for v in p.values()]},fp,mode="w")
    rebuild_labels()

# ────────── Flask routes  ─────────────────────────────────────────
app=Flask(__name__)

@app.route("/")
def index(): return render_template("index.html",dataset_len=DS_LEN)

def payload(idx:int):
    samp=DATASET[idx]; rgb=tens_to_rgb(samp["front_rgb"])
    ann=ANN.get(idx,{"paths":{}})
    path_png = mask_to_png((samp["path_mask"].cpu().numpy().squeeze()>0.5),
                           (255,255,51),120)
    out={"front_rgb":png_bytes(rgb),"path_mask_png":path_png,"paths":{}}
    for pid,v in ann["paths"].items():
        mpng = (mask_to_png(v["mask"],colour_for_pid(pid),128)
                if v["mask"] is not None else "")
        out["paths"][pid]={"spline":(v["spline"].tolist() if v["spline"] is not None else []),
                           "goal":v["goal"],"pts":v["pts"],
                           "mask_png":mpng}
    return out

@app.route("/sample",methods=["POST"])
def sample():
    idx=max(0,min(DS_LEN-1,int(request.json["idx"])))
    if idx not in ANN:
        seq=DATASET[idx]["infos"]["sequence"]; frame=DATASET[idx]["infos"]["frame"]
        ANN[idx]=load_ann(ann_fp(seq,frame))
    return jsonify(payload(idx))

@app.route("/update_mask",methods=["POST"])
def update_mask():
    global SAM2
    d=request.get_json(); idx=int(d["idx"]); pid=str(d["pid"]); pts=d["pts"]
    ann=ANN.setdefault(idx,{"paths":{}})
    if pid=="new" or pid not in ann["paths"]:
        pid=str(len(ann["paths"]))
        ann["paths"][pid]={"spline":None,"goal":"","mask":None,"pts":[]}
    rgb=tens_to_rgb(DATASET[idx]["front_rgb"])
    mnew=compute_sam2_mask(SAM2, rgb, pts, device=DEVICE)
    cur=ann["paths"][pid]["mask"]
    merged=mnew if cur is None else np.logical_or(cur,mnew)
    ann["paths"][pid]["mask"]=merged
    ann["paths"][pid]["pts"].extend(pts)
    return jsonify({"pid":pid,"mask_png":mask_to_png(merged,colour_for_pid(pid),128)})

@app.route("/add_path",methods=["POST"])
def add_path():
    d=request.get_json(); idx=int(d["idx"]); pid=str(d.get("pid","new"))
    ann=ANN.setdefault(idx,{"paths":{}})
    if pid=="new" or pid not in ann["paths"]:
        pid=str(len(ann["paths"]))
        ann["paths"][pid]={"spline":None,"goal":"","mask":None,"pts":[]}
    ann["paths"][pid]["spline"]=spline_fit(d["pts"])
    ann["paths"][pid]["goal"]=d.get("goal","")
    return jsonify({"paths":payload(idx)["paths"]})

@app.route("/delete_mask",methods=["POST"])
def del_mask():
    idx=int(request.json["idx"]); pid=str(request.json["pid"])
    if idx in ANN and pid in ANN[idx]["paths"]:
        ANN[idx]["paths"][pid]["mask"]=None; ANN[idx]["paths"][pid]["pts"]=[]
    return jsonify({"status":"cleared"})

@app.route("/delete_path",methods=["POST"])
def del_path():
    idx=int(request.json["idx"]); pid=str(request.json["pid"])
    if idx in ANN:
        ANN[idx]["paths"].pop(pid,None)
        # renumber
        new={}
        for i,(_,v) in enumerate(sorted(ANN[idx]["paths"].items(),key=lambda kv:int(kv[0]))):
            new[str(i)]=v
        ANN[idx]["paths"]=new
    return jsonify({"paths":payload(idx)["paths"]})

@app.route("/save",methods=["POST"])
def save():
    idx=int(request.json["idx"])
    seq=DATASET[idx]["infos"]["sequence"]; frame=DATASET[idx]["infos"]["frame"]
    fp=ann_fp(seq,frame)
    if idx not in ANN or not ANN[idx]["paths"]:
        fp.unlink(missing_ok=True); prune_dirs(fp.parent); rebuild_labels()
        return jsonify({"error":"nothing to save"}),400
    save_ann(fp,ANN[idx])                  # rebuild_labels() inside save_ann
    return jsonify({"saved":"ok","file":str(fp)})

if __name__=="__main__":
    app.run(host=ARGS.host, port=ARGS.port, debug=True, threaded=False)
