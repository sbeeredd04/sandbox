# build_model.py
import yaml, importlib
from pathlib import Path
import pytorch_lightning as pl
import torch                   # only for type hints; no direct use

from omegaconf import DictConfig, OmegaConf

from scripts.utils.log_utils import logging
from spinflow.model.flowpolicy import WaypointFlowPolicy
from spinflow.model.lelan.lelan import LeLaN_clip
from spinflow.model.marigold import MarigoldModel

def build_model(cfg: DictConfig, ckpt_path: str, vision_ckpt_path=None, seed=42, device="cuda"):
    # cfg = DictConfig(cfg)
    assert "model_name" in cfg, \
        "Model config must contain 'model_name' to import the model class."

    target = cfg.model_name
    Model = globals().get(target, None)
    if Model is None:
        logging.error(f"Model {target} not found in the current module.")
        raise ValueError(f"Model {target} not found. Available models: {list(globals().keys())}")
    
    model = Model(cfg).to(device)
    if ckpt_path:
        # 2. load only the "state_dict" part on CPU
        sd = torch.load(
            ckpt_path, 
            mmap=True,
            weights_only=False, 
            map_location="cpu"
        )["state_dict"]
        # optionally strip "model." prefixes saved by PL
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}

        # Optionally override the weights for the vision encoder
        if vision_ckpt_path is not None:
            vision_sd = torch.load(
                vision_ckpt_path, 
                mmap=True,
                weights_only=False, 
                map_location="cpu"
            )["state_dict"]
            vision_sd = {k.replace("model.", "", 1): v for k, v in vision_sd.items()}

            # Add vision_encoder. prefix to model keys
            vision_sd = {f"vision_encoder.{k}": v for k, v in vision_sd.items()}
            logging.info(f"Replace {len(vision_sd)} vision encoder weights from {vision_ckpt_path}")
            for k, v in vision_sd.items():
                # if k in sd:
                #     print(f"Overriding weight for {k}")
                sd[k] = v

        missing, unexpected = model.load_state_dict(
            sd, strict=False
        )

        # assert len(missing) == 0, f"Missing keys: {missing}"
        logging.info(
            f"Loaded {ckpt_path} → {target}  "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        model = model.to(device)
        logging.info(f"Loaded model {target} from checkpoint {ckpt_path}")
    else:
        logging.warning(f"No checkpoint provided, using untrained model {target}")
        
    return model