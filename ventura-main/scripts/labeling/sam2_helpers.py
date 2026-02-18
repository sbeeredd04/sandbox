import torch
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor

def create_sam2(device="cuda")->SAM2ImagePredictor:
    mdl = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    mdl.model.to(device).eval(); return mdl

def compute_sam2_mask(sam2, rgb:np.ndarray,pts:list[list[float]], device="cuda")->np.ndarray:
    if not pts: return np.zeros(rgb.shape[:2],bool)
    sam2.set_image(rgb)
    coords=np.asarray(pts,np.float32)
    labels=np.ones(len(coords),np.int32)
    ctx = (torch.autocast("cuda",dtype=torch.bfloat16)
           if device=="cuda" else torch.inference_mode())
    with torch.inference_mode(), ctx:
        masks,_,_=sam2.predict(point_coords=coords,
                               point_labels=labels,
                               multimask_output=False)
    return masks[0].astype(bool)