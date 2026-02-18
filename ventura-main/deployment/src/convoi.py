
"""
Convoi-esque GPT reimplementation 

Assumes that inputs are not synchronized, convoi will handle explicitly handle
input synchronization between camera + LiDAR inputs.
"""
import torch
import torch.nn as nn

import logging
import copy
import cv2
import numpy as np
from dotenv import load_dotenv
from deployment.src.motion_templates import (
    MotionTemplateLibrary,
    project_xyz_to_uv,
)
from deployment.src.gpt4v import (
    BaseO4Mini
)
from deployment.src.constants import (
    CAM_TO_BASE_OFFSET
)

from scripts.utils.log_utils import logging

def _visible_remap_left_to_right(uv_all: np.ndarray,
                                 valid_mask: np.ndarray):
    """
    Returns:
      uv_draw  : (M,2) UVs of visible endpoints, ordered left→right
      draw2arc : (M,)  map from drawn index → original arc index
    """
    vis_idx = np.flatnonzero(valid_mask)
    if vis_idx.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int64)
    order   = np.argsort(uv_all[vis_idx, 0])          # sort by u (x) ascending
    draw2arc = vis_idx[order]
    uv_draw  = uv_all[draw2arc]
    return uv_draw, draw2arc

class ConvoiPlannerNode(nn.Module):
    def __init__(self, model_cfg, robot_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.robot_cfg = robot_cfg

        # --- Load environment variables from .env file
        load_dotenv()
        logging.info("Loaded environment variables from .env file")

        # --- Initialize the planner node
        self.input_keys = [pair['out_key'] for pair in model_cfg['dataloader_inputs']]
        self.num_actions = model_cfg['num_actions']
        self.action_dim = model_cfg['action_dim']

        # --- Initialize physical and sampler params
        self.nav_params = robot_cfg['navigation_parameters']
        self.sampler_params = robot_cfg['sampler_parameters']

        self.mt_cfg = self.sampler_params  # {num_options, max_curvature, max_free_path_length}

        model_name = 'gpt-4o-mini-2024-07-18'
        logging.info(f"Using VLM model: {model_name}")
        self.vlm = BaseO4Mini("OPENAI_API_KEY", model_name=model_name)

        self.default_idx = 0

    def default_outputs(self):
        return {
            "action_pred": torch.zeros((1, self.num_actions, self.action_dim), dtype=torch.float32)
        }

    def motion_templates(self):
        mt_cfg = copy.deepcopy(self.mt_cfg)
        motion_bank = MotionTemplateLibrary(
            max_curvature=float(mt_cfg['max_curvature']),
            max_path_len=float(mt_cfg['max_free_path_length']),
            num_options=int(mt_cfg['num_options'])
        )
        return motion_bank.arcs()

    def get_prompt(self, instruction: str):
        return f"""
I am a wheeled robot. This is the image I'm seeing right now. I have annotated it with numbered circles. Each number represent a general direction I can follow. You are a five-time world-champion navigation agent and your task is to tell me which circle I should pick for the task of: {instruction}?
You MUST CHOOSE EXACTLY {1} best candidate number. Avoid choosing routes that go through objects and untraversable terrains and regions. Skip analysis and provide your answer at the end in a json file of this form:
{{"points": [] }}
"""
    
    def annotate_image(self, img: np.ndarray, pts: np.ndarray, selected_idx=None):
        h, w = img.shape[:2]
        overlay = img.copy()
        circle_rad = 10

        annotation_dict = {}
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5                    # tweak if you change circle_rad
        thickness = 1
        text_color = (0, 0, 0)             # black

        for i, p in enumerate(pts):
            annotation_dict[i] = p
            cx, cy = int(p[0]), int(p[1])

            # circle (green if selected, else white)
            color = (0, 255, 0) if (selected_idx is not None and i == selected_idx) else (255, 255, 255)
            cv2.circle(img, (cx, cy), circle_rad, color, -1)

            # measure text and center it
            label = str(i)
            (tw, th), baseline = cv2.getTextSize(label, font, fontScale, thickness)
            # x,y is bottom-left corner for putText
            tx = cx - tw // 2
            ty = cy + th // 2

            # (optional) outline for legibility
            cv2.putText(img, label, (tx, ty), font, fontScale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(img, label, (tx, ty), font, fontScale, text_color, thickness, cv2.LINE_AA)

        # light blend to keep things readable
        alpha = 0.3
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return img, annotation_dict
        
    def infer(self, inputs, **kwargs):
        outputs = self.default_outputs()
        for key in self.input_keys:
            if key not in inputs:
                print(f"Missing input key: {key}, skipping inference.")
                return outputs
        if 'intrinsics' not in inputs:
            print("Missing camera intrinsics, skipping inference.")
            return outputs

        # 1) Build arcs and endpoints (original order: curvature -max → +max)
        arcs = self.motion_templates()
        curvatures = np.array([a.curvature for a in arcs], dtype=np.float32)
        endpts_xy = np.array([a.xy_at_s(a.length) for a in arcs], dtype=np.float32)
        endpts_xyz = np.concatenate([endpts_xy, np.full((len(endpts_xy), 1), -0.4, np.float32)], axis=1)

        # 2) Project *all* endpoints, keep mask
        uv_all, vis_mask = project_xyz_to_uv(
            xyz=endpts_xyz,
            intrinsics=inputs['intrinsics'],
            T_base_to_optical=inputs.get('T_base_to_optical', None),
        )

        # 3) Remap visible ones to contiguous labels 0..M-1 in curvature order
        uv_draw, draw2arc = _visible_remap_left_to_right(uv_all, vis_mask)

        # 4) Draw only visible points, labeled 0..M-1
        rgb_image = inputs['rgb_image'].permute(0, 2, 3, 1).cpu().numpy().copy()
        rgb_image_np = np.clip((rgb_image[0] + 1) * 0.5, 0, 1)
        rgb_image_np = (rgb_image_np * 255.0).astype(np.uint8)
        H, W = inputs['intrinsics']['image_height'], inputs['intrinsics']['image_width']
        rgb_image_np = cv2.resize(rgb_image_np, (W, H), interpolation=cv2.INTER_LINEAR)

        tmp_img = rgb_image_np.copy()
        tmp_img, _ = self.annotate_image(tmp_img, uv_draw, selected_idx=None)
        cv2.imwrite("tmp_convoi.jpg", tmp_img)
        outputs['annotated_img'] = tmp_img

        # 5) VLM choice
        goal_prompt = self.get_prompt(inputs['goal_command'][0])
        result = self.vlm.generate_text_individual(
            text=goal_prompt,
            image_filepaths=["tmp_convoi.jpg"],
        )
        if len(result) == 0:
            print("No result returned from VLM.")
            return outputs

        try:
            print("result ", result)
            choice = self.vlm.parse_response(result)[0]
            print("choice ", choice)
        except Exception as e:
            logging.error(f"Failed to parse VLM response: {e}")
            return outputs

        # 6) Map from drawn label → arc index, then use that arc
        if len(draw2arc) == 0:
            logging.warning("No visible arcs; returning zeros.")
            return outputs
        elif choice >= len(draw2arc):
            logging.warning(f"VLM chose index {choice} out of bounds of {len(draw2arc)}")
        arc_idx = int(draw2arc[choice])      # <- original arc index
        selected_arc = arcs[arc_idx]

        # 7) Build waypoints along that arc
        s_range = np.linspace(0, selected_arc.length, self.model_cfg['num_actions'])
        waypoints_xy = np.array([selected_arc.xy_at_s(s) for s in s_range], dtype=np.float32)
        waypoints_xyz = np.concatenate([waypoints_xy, np.full((len(waypoints_xy), 1), -0.4, np.float32)], axis=1)

        outputs['action_pred'] = torch.from_numpy(waypoints_xyz.reshape(1, -1, self.action_dim)).float().to("cuda")
        # outputs['action_pred'][:, :, :] += torch.tensor(CAM_TO_BASE_OFFSET).to(device) # Transform to CAM frame


        # (optional) re-draw with selected index highlighted
        annotated_img = rgb_image_np.copy()
        annotated_img, _ = self.annotate_image(annotated_img, uv_draw, selected_idx=choice)
        outputs['annotated_img'] = annotated_img

        # Keep the mapping if you need it outside:
        outputs['draw2arc'] = draw2arc  # numpy array mapping drawn label → original arc index
        return outputs