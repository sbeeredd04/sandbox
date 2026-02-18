#!/usr/bin/env python3
import argparse, base64, cv2, io, yaml, torch
import numpy as np
from pathlib import Path
from flask import Flask, jsonify, render_template, request

import hydra
from omegaconf import DictConfig, OmegaConf

from build_dataset import build_dataset
from build_model    import build_model

from torchvision.transforms import ToTensor
from scripts.utils.satellite_utils import (gps_to_local_xy, annotate_satellite_image)
# from scripts.labeling.sam2_helpers import (create_sam2, compute_sam2_mask)
from spinflow.util.vis_utils import (
    draw_xyz_on_image
)

# ───── global constants ────────────────────────────────────────────
NUM_STEPS = 25
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
GEN       = torch.Generator(device=DEVICE)
# SAM2      = create_sam2(DEVICE)

# ───── CLI ─────────────────────────────────────────────────────────
# pa = argparse.ArgumentParser()
# pa.add_argument("--dataset_cfg", default="config/dataset/planning/frodo8k_turn_goal.yaml")
# pa.add_argument("--model_cfg",   required=True)
# pa.add_argument("--weights",     required=True)
# pa.add_argument("--split_dir", required=False, help="Override dataset split subdirectory (e.g. divergence_splits)")
# pa.add_argument("--host", default="0.0.0.0")
# pa.add_argument("--port", type=int, default=5000)
# args = pa.parse_args()

# ───── helpers ─────────────────────────────────────────────────────
def load_cfg(p):         # YAML → dict
    with open(p) as f: return yaml.safe_load(f)

def get_key_cfg(cfg_root:dict, name:str):
    """search top-level & sub-keys (dict type)"""
    for item in cfg_root.get("load_cfgs", []):
        if item.get("name") == name: return item
        if item.get("type") == "dict":
            for sub in item["kwargs"].get("subkeys", []):
                if sub.get("name") == name: return sub
    return {}

def tens_to_rgb(t):
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return cv2.normalize(arr,None,0,255,cv2.NORM_MINMAX).astype("uint8")

def mask_to_tens(mask):
    """Convert a mask (H,W) to a tensor (1,1,H,W) with float32 dtype."""
    mask = (mask.astype(np.float32) * 2.0) - 1.0  # scale to [-1,1] for ControlNet 
    return torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def png_bytes(rgb):
    return base64.b64encode(
        cv2.imencode(".png", cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))[1]
    ).decode()

def blend(rgb, mask):
    out = rgb.copy()
    m   = (mask>0.5).cpu().numpy()[0,0]
    out[m] = (51,255,255)
    return cv2.addWeighted(rgb,0.5,out,0.5,0)

# ───── factory ─────────────────────────────────────────────────────
def create_app(cfg)->Flask:
    ds_cfg  = cfg['dataset'] #load_cfg(cfg['dataset'])
    mdl_cfg = cfg['model'] #load_cfg(cfg['model'])
    if "scheduler" in mdl_cfg["validation"]:
        mdl_cfg["validation"]["scheduler"]["kwargs"]["default_denoising_steps"] = NUM_STEPS

    dataset = build_dataset(ds_cfg)
    assert "weights_ckpt" in mdl_cfg, \
        "Model config must contain 'weights_ckpt' for inference."
    model   = build_model(mdl_cfg, mdl_cfg['weights_ckpt'], seed=42, device=DEVICE).eval()

    # Get pipeline
    if "pipeline" in mdl_cfg:
        pipeline = mdl_cfg['pipeline']
    else:
        pipeline = mdl_cfg["vision_encoder"]["pipeline"]
    
    if "unet" in pipeline:
        # Unet pipeline is used for ControlNet
        input_keys = pipeline["unet"]["input_keys"]
    elif "vision_encoder" in pipeline:
        input_keys = pipeline["vision_encoder"]["input_keys"]


    # schema drives GUI panels
    mod_map = {k["name"]:k["modality"]
               for k in input_keys}
    schema  = [{"in_key":d["in_key"],
                "modality": mod_map.get(d["out_key"], d["out_key"]),}
               for d in mdl_cfg["dataloader_inputs"]]

    app   = Flask(__name__)
    cache = {"idx":None, "samp":None, "click_pts":[]}

    # ─── internal helpers ──────────────────────────────────────────
    def make_payload(samp, pred=None):
        rgb_base = tens_to_rgb(samp["front_rgb"])
        cv2.imwrite("inference_front_rgb.png", cv2.cvtColor(rgb_base, cv2.COLOR_RGB2BGR))
        # union of GT path + user target segment (if any)
        overlay = samp["path_mask"].clone()
        if "target_segment" in samp:
            overlay = (overlay > .5) | (samp["target_segment"] > .5)
            overlay = overlay.float()

        data = {
            "meta": dict(sequence=samp["infos"]["sequence"],
                         frame   =int(samp["infos"]["frame"]),
                         idx     =int(samp["infos"].get("idx",-1))),
            # GT overlay replaces plain front_rgb
            "front_rgb": png_bytes(blend(rgb_base, overlay)),
            "robot_gps":    samp["current_gps"].tolist() if "current_gps" in samp else None,
            "goal_gps":     samp["future_gps"][0].tolist() if "future_gps" in samp else None,
            "robot_heading": float(samp["current_heading"]) if "current_heading" in samp else None,
        }

        # add schema-listed items
        for itm in schema:
            k = itm["in_key"]
            if k in data:                      # already filled (e.g. blended rgb)
                continue
            if k not in samp: continue
            if itm["modality"]=="image":
                data[k] = png_bytes(tens_to_rgb(samp[k]))
            else:
                data[k] = str(samp[k])
        if pred is not None:
            if "path_mask_pred" in pred:
                pred_mask = pred["path_mask_pred"]
                ann_rgb = blend(rgb_base, pred_mask)
            else:
                ann_rgb = rgb_base.copy()

            # Project action prediction to image
            action_dim = pred['action_pred'].shape[-1] if "action_pred" in pred else 3
            offset = torch.tensor([0, 0, 0.0], device=DEVICE).view(1, 1, action_dim)
            if "action_pred" in pred:
                # Manual offset to z value of action prediction
                action_pred = pred["action_pred"]
                action_pred = action_pred + offset.to(action_pred.device)
                ann_rgb = draw_xyz_on_image(
                    ann_rgb,
                    action_pred,
                    samp['infos']
                )
            if "action_label" in samp:
                action_label = samp["action_label"]
                if action_label.ndim == 2:
                    action_label = action_label.unsqueeze(0)
                action_label = action_label + offset.to(action_label.device)
                ann_rgb = draw_xyz_on_image(
                    ann_rgb,
                    action_label,
                    samp['infos'],
                    color=(0, 0, 255)  # blue for label
                )
            data["front_rgb_pred"] = png_bytes(ann_rgb)

        return data

    def model_inputs(samp):
        out={}
        for d in mdl_cfg["dataloader_inputs"]:
            v=samp[d["in_key"]]
            if isinstance(v, str):
                out[d["out_key"]] = [v]
            else:
                ndim = v.ndim
                if ndim <= 4:  # (C,H,W) → (1,C,H,W)
                    v = torch.as_tensor(v).unsqueeze(0)
                v = v.to(device=DEVICE, dtype=torch.float32)
                out[d["out_key"]] = v
        return out

    # ─── routes ────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html",
                               dataset_len=len(dataset),
                               input_schema=schema)

    @app.route("/sample", methods=["POST"])
    def sample():
        idx=int(request.json["idx"])
        cache.update(idx=idx, samp=dataset[idx], click_pts=[])
        return jsonify(make_payload(cache["samp"]))

    @app.route("/predict", methods=["POST"])
    def predict():
        req = request.json
        idx = int(req["idx"])
        samp = cache["samp"] if cache["idx"] == idx else dataset[idx]

        # apply in-browser text edits (unchanged) …
        for k, txt in req.get("texts", {}).items():
            if k in samp and isinstance(samp[k], str):
                samp[k] = txt

        # cfg_scale = float(req.get(
        #     "cfg_scale",
        #     mdl_cfg["validation"]["scheduler"]["kwargs"].get("cfg_scale", 1.0)
        # ))
        # mdl_cfg["validation"]["scheduler"]["kwargs"]["cfg_scale"] = cfg_scale
        # guidance_rescale = mdl_cfg["validation"]["scheduler"]["kwargs"].get(
        #     "guidance_rescale", 0.0
        # )
        with torch.no_grad():
            if mdl_cfg['model_name'] == 'MarigoldModel':
                scheduler_kwargs = mdl_cfg['validation']['scheduler']['kwargs']
                preds = model.infer(
                    model_inputs(samp),
                    num_inference_steps=scheduler_kwargs['default_denoising_steps'], 
                    cfg_scale=scheduler_kwargs.get('cfg_scale', 8.0),
                    guidance_rescale=scheduler_kwargs.get('guidance_rescale', 0.5),
                    generator=GEN,
                    show_progress_bar=True
                )
                preds['path_mask_pred'] = preds['target_pred']  # for compatibility
            else:
                preds = model.infer(
                    model_inputs(samp),
                    **mdl_cfg['validation']['integrator']['kwargs']
                )

        cache.update(idx=idx, samp=samp)
        return jsonify(make_payload(samp, preds))

    @app.route("/set_goal", methods=["POST"])
    def set_goal():
        req = request.json
        idx = int(req["idx"])
        lat = float(req["lat"]); lon = float(req["lon"])
        if not (0<=idx<len(dataset)):
            return jsonify({"error":"index out of range"}),400
        samp = cache["samp"] if cache["idx"]==idx else dataset[idx]
        new  = dataset.apply_new_goal(samp, lat, lon)
        cache.update(idx=idx, samp=new)
        return jsonify(make_payload(new))

    @app.route("/upload_image", methods=["POST"])
    def upload_image():
        req  = request.json
        idx  = int(req["idx"]); key=req["key"]; b64=req["b64"]
        if not (0<=idx<len(dataset)):
            return jsonify({"error":"idx out of range"}),400
        samp = cache["samp"] if cache["idx"]==idx else dataset[idx]

        buf = base64.b64decode(b64)
        img = cv2.imdecode(np.frombuffer(buf,np.uint8),cv2.IMREAD_COLOR)
        if img is None: return jsonify({"error":"decode"}),400
        img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # If the user drops a new front_rgb we want to keep its tensor size
        if key == "front_rgb" and key in samp:
            # samp['front_rgb'] shape is (1,3,H,W)
            _, _, H, W = samp[key].shape
            dims = [3, H, W]
            rng  = [-1, 1]                      # ControlNet expects [-1,1] range
        else:
            cfg_k = get_key_cfg(ds_cfg,key)
            dims  = cfg_k.get("dimensions",[3,*img_rgb.shape[:2][::-1]])
            rng   = cfg_k.get("range",[-1,1])

        img_rgb=cv2.resize(img_rgb,(dims[2],dims[1]),cv2.INTER_AREA)

        dev  = samp[key].device if key in samp and torch.is_tensor(samp[key]) else torch.device("cpu")
        dtype= samp[key].dtype  if key in samp and torch.is_tensor(samp[key]) else torch.float32

        t = ToTensor()(img_rgb).unsqueeze(0).to(device=dev,dtype=torch.float32)
        lo,hi=rng
        t = t*(hi-lo)+lo
        samp[key]=t.to(dtype=dtype)
        cache.update(idx=idx, samp=samp)
        return jsonify({"status":"ok"})
    
    @app.route("/add_click", methods=["POST"])
    def add_click():
        """Receive a normalised click (x,y in [0,1]) on front_rgb."""
        req  = request.json
        idx  = int(req["idx"]);  x = float(req["x"]);  y = float(req["y"])
        if cache["idx"] != idx:
            return jsonify({"error":"stale idx"}), 400

        samp = cache["samp"]
        H, W = samp["front_rgb"].shape[-2:]          # (1,3,H,W)
        px   = [x * W, y * H]                        # SAM-2 wants pixel coords

        cache["click_pts"].append(px)

        # --- run SAM-2 once on the accumulated points ---
        rgb_np = tens_to_rgb(samp["front_rgb"])      # HxWx3 uint8
        # mask   = compute_sam2_mask(SAM2, rgb_np, cache["click_pts"],
        #                            device=DEVICE).astype(np.float32)
        mask = np.zeros((H, W), dtype=np.float32)  # placeholder for SAM-2 output
        tsr    = mask_to_tens(mask)
        samp["target_segment"] = tsr

        cache["samp"] = samp          # write-back
        return jsonify(make_payload(samp))   # refreshed overlay

    return app

@hydra.main(
    version_base="1.3",
    config_path="../../config",
    config_name="policy",
)
def main(cfg: DictConfig):
    """
    Hydra will read config/app.yaml, 
    giving you:
      cfg.dataset_cfg:   path/to/planning/frodo8k_turn_goal.yaml
      cfg.model_cfg:     path/to/simple_policy.yaml
      cfg.weights:       path/to/ckpt.ckpt
      cfg.split_dir:     e.g. divergence_splits
      cfg.host, cfg.port
      cfg.device
      cfg.seed
    """
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Autoinject host, port, device, and seed
    OmegaConf.set_struct(cfg, False)
    if "host" not in cfg:
        cfg.host = "0.0.0.0"
    if "port" not in cfg:
        cfg.port = 5000
    if "seed" not in cfg:
        cfg.seed = 42
    if "device" not in cfg:
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    OmegaConf.set_struct(cfg, True)

    flask_app = create_app(cfg)
    flask_app.run(host=cfg.host, port=cfg.port, debug=True)

# ───── run ─────────────────────────────────────────────────────────
if __name__=="__main__":
    main()