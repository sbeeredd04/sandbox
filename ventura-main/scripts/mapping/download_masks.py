"""
This file contains all processing scripts necessary for computing
target masks for navigation.
"""

import torch
import os
import random
import numpy as np
from PIL import Image
import decord
import cv2
import supervision as sv
from tqdm import tqdm
import hickle as hkl
from pathlib import Path
import pandas as pd
import json

# print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
# print("torch.version.cuda:", torch.version.cuda)
# print("nvcc path:", os.popen("which nvcc").read().strip())

# Qwen imports
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel

# blip3o imports
from qwen_vl_utils import process_vision_info
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from blip3o.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

# Grounded-SAM imports
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

from scripts.utils.log_utils import logging
from spinflow.util.path_utils import (
    get_closest_entity_mask,
    blend_mask
)
from spinflow.dataset.frodo_helpers import (
    set_frodo_dir
)

# QWEN_PROMPT = """Describe objects that are close in the image that the viewer is navigating towards.
# Do not describe the ground or objects that are in the background.
# Do not describe objects that are far away or not relevant for navigation.
# Output the nouns in a comma separated list format."""

# QWEN_PROMPT = """Describe objects that are nearby in the image that the viewer is navigating towards.
# Some interesting examples of objects are cars, doors, trash cans, people, fire hydrants, etc.
# Describe objects that would be common landmarks for navigation.
# Do not describe the ground unless is semantically important for navigation like crosswalk.
# Output the nouns in a comma separated list format."""

# Results in too general descriptions not important for navigation
# QWEN_CAPTIONING_PROMPT = """
# Describe the terrain that the robot is staying on (e.g. sidewalk, grass, gravel, brick pathway, stairs, etc.).
# Describe the entities that the robot is navigating directly to (e.g. stairs, door, trash can, driveway, watering hose).
# The generated text should be in the format:
# <action> on <staying_on_terrain>,
# <action> the <avoiding terrain>,
# <action> to <navigating_to_entity1>, <navigating_to_entity2>, ... 
# Vary the <action> keyword with the appropriate robot action based on the context.
# Do not output any other text.
# """

# QWEN_CAPTIONING_PROMPT = """
# Describe unique and specific objects and landmarks that are closest
# to the end of the aqua blue swept volume in the image.
# Do not describe terrains like road, street, and sidewalk. 
# Output the nouns in a comma separated list format.
# If no objects are visible in the image, output an empty string.
# """
QWEN_CAPTIONING_PROMPT = """
The prompt below describes the future behavior of the robot in the image. Generate 10 variations of the 
the prompt that seem more natural and human-like. The variations should still semantically and spatially
match the original prompt exactly. For instance, if the original prompt says "Drive around the tree from the right side", 
then the variation should not change the direction, but can change the action "drive" to "move"
Output the 10 variations in a comma separated list format. Below is an example of the inputs and outputs:
Example:
Input: "Drive around the tree from the right side"
"Drive around the tree from the right side"
Output:
"Move around the tree from the right side, Pass the tree from the right, Navigate around the tree staying on the right, Keep to the right while avoiding the tree, Go around the tree from the right side, Circumnavigate the tree from the right, Keep to the right and move
"""

OPENAI_CAPTIONING_PROMPT = """
You are a helpful assistant that understands an image and a caption describing the future behavior of a robot taking the image.
Your task is to generate 10 variations of the caption that are natural and contextually relevant, while maintaining the original meaning and spatial relationships described in the caption.
You should provide the variations in a json file of this form:
{{"scene_captions": [<caption variation 1>, <caption variation 2>, ..., <caption variation 10>]}}
"""


PROCESS_CAPTION_ONLY = True
PERFORM_TRACKING = False
OBJECT_SELECTION_METHOD = "distance" # "vlm"

def process_image(
    prompt: str, 
    img: Image.Image, 
    vlm_model_dict: dict,
    # processor: AutoProcessor,
    # multi_model, 
    device: str = "cuda"
) -> str:
    if vlm_model_dict["model_type"] == "openai":
        # Save image to unique file name even with multiple identical processes
        model = vlm_model_dict["multi_model"]
        model_result = None
        num_tries = 0

        full_prompt = OPENAI_CAPTIONING_PROMPT + f"Caption: {prompt.strip()}"
        while model_result is None and num_tries < 10:
            try:
                json_response = model.generate_text_individual(
                    text=full_prompt,
                    image_filepaths=[img],
                )
                model_result = model.parse_response(json_response, expected_key="scene_captions")
                if len(model_result) < 10:
                    raise json.JSONDecodeError("Not enough caption variations generated")
            except json.JSONDecodeError:
                print(f"Attempt {num_tries + 1}: Failed to parse JSON response. Retrying...")
                model_result = None
                num_tries += 1
        
        if model_result is None:
            return []  # Return empty list if parsing fails after retries
        return model_result
    elif vlm_model_dict["model_type"] == "blip3o":
        processor = vlm_model_dict["processor"]
        multi_model = vlm_model_dict["multi_model"]

        caption_prompt = QWEN_CAPTIONING_PROMPT + "Prompt: " + prompt.strip()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": caption_prompt},
            ],
        }]
        text_prompt_for_qwen = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt_for_qwen],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        generated_ids = multi_model.generate(**inputs, max_new_tokens=1024)
        input_token_len = inputs.input_ids.shape[1]
        generated_ids_trimmed = generated_ids[:, input_token_len:]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return output_text
    raise ValueError(f"Unsupported model type: {vlm_model_dict['model_name']}")

def segment_masks(
    image_np: np.ndarray,
    prompt: str,
    image_predictor: SAM2ImagePredictor,
    grounding_model: AutoModelForZeroShotObjectDetection,
    processor: AutoProcessor,
    device: str = "cuda"
):
    """
    Detects masks in a single image based on the provided prompt.
    
    Args:
        image (Image.Image): The input image.
        prompt (str): The prompt for mask detection.
        video_predictor (SAM2ImagePredictor): The SAM2 image predictor.
        grounding_model (AutoModelForZeroShotObjectDetection): The grounding model for object detection.
        dino_processor (AutoProcessor): The processor for the grounding model.
        device (str): The device to run the model on ("cuda" or "cpu").
    
    Returns:
        dictionary: A dictionary containing the detected masks and their bounding boxes.
    """
    img = Image.fromarray(image_np).convert("RGB")  # Convert to PIL Image
    if "," in prompt:
        prompt = prompt.replace(",", ".")

    #2 Get bbox detection seeds
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        # Frodo thresholds
        # box_threshold=0.4,
        # text_threshold=0.4,
        # FAI thresholds
        box_threshold=0.2,
        text_threshold=0.3,
        target_sizes=[img.size[::-1]]
    )
    image_predictor.set_image(np.array(img.convert("RGB")))

    if len(results) == 0:
        print("No objects detected in the image.")
        return None

    input_boxes = results[0]["boxes"].cpu().numpy()
    objects = results[0]["labels"]  

    if len(input_boxes) == 0:
        print("No bounding boxes detected in the image.")
        return None

    #3 Extract SAM2 masks
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)  # Add batch dimension
    if scores.ndim == 2:
        scores = scores.squeeze(1)  # Add batch dimension

    return {
        "masks": masks,
        "scores": scores,
        "logits": logits,
        "boxes": input_boxes,
        "labels": objects,
    }

def track_masks(
    video_np: np.ndarray,
    objects: list,
    masks: np.ndarray,
    video_path: str,
    video_predictor: SAM2VideoPredictor,
):
    inference_state = video_predictor.init_state(video_path)
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1

    PROMPT_TYPE_FOR_VIDEO = "mask"
    for object_id, (label, mask) in enumerate(zip(objects, masks), start=1):
        labels = np.ones((1), dtype=np.int32)
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            mask=mask
        )

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    return video_segments

def draw_masks_single(
    image: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    labels,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Draws masks, bounding boxes, and labels on a single RGB image using supervision.

    Args:
        image:  H×W×3 uint8 RGB image.
        masks:  (N, H, W) bool or float mask array.
        boxes:  (N, 4) array of [xmin, ymin, xmax, ymax] in pixel coords.
        labels: length-N list of class names (strings).
        alpha:  mask transparency (0–1).

    Returns:
        Annotated H×W×3 uint8 image.
    """
    img = image.copy()

    # Ensure mask is boolean
    masks_bool = masks > 0.5 if masks.dtype != bool else masks
    N = masks_bool.shape[0]

    # Convert boxes to int pixel coords
    boxes_int = boxes.astype(int)

    # Give each detection a numeric ID 1..N
    class_ids = np.arange(1, N + 1, dtype=np.int32)

    # Build a supervision Detections object
    detections = sv.Detections(
        xyxy=boxes_int,
        mask=masks_bool,
        class_id=class_ids,
    )

    # Create the annotators
    mask_annotator  = sv.MaskAnnotator()
    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    
    # Draw in this order: masks → boxes → labels
    out = mask_annotator.annotate(scene=img,                        detections=detections)
    out = box_annotator.annotate(scene=out,                         detections=detections)
    out = label_annotator.annotate(scene=out, detections=detections, labels=list(labels))

    return out

def compute_entity_masks(
    ride_name: str,
    start_frame: int,
    end_frame: int,
    root_dir: str,
    out_dir: str,
    vlm_model_dict: dict,
    dino_model_dict: dict,
    sam2_model_dict: dict,
    save_visualizations: bool = False,
    caption_only: bool = False,
    cache_enabled: bool = True,
    device: str = "cuda"
):

    """
    Computes target masks for a given ride using Qwen2.5, Grounding DINO, and SAM2 models.
    """
    parts = ride_name.split(' ')
    ride_dir = set_frodo_dir(out_dir.parent, *parts)  # e.g. "ride_1234_5678_91011_20250101T123000Z"
    assert len(parts) >= 4, f"Cannot parse ride_name={ride_name}"
    ride_id = parts[0]
    driveid0 = parts[1]
    driveid1 = parts[2]
    timestamp = "_".join(parts[3:])
    split_file = ride_dir.parent / "full_raw.txt"
    assert split_file.exists(), f"Split file {split_file} does not exist."
    split_df = pd.read_csv(split_file, sep=",", header=0)
    # Grab row with matching ride_name and start_frame
    match_row = split_df[(split_df["ride_name"] == ride_name)]
    if match_row.empty:
        print(f"No matching row found for ride {ride_name} and start frame {start_frame}.")
        return False

    # Get the label such that the start_frame is within the start/end frame range
    frame_ranges = np.vstack((match_row["start_frame"].values, match_row["end_frame"].values)).T
    frame_ranges = np.sort(frame_ranges, axis=0)  # Sort by start_frame

    # Binary search for the correct frame range index
    split_row = None
    for i, (start, end) in enumerate(frame_ranges):
        if start <= start_frame <= end:
            split_row = match_row.iloc[i]
            break

    if split_row is None:
        print(f"No matching frame range found for ride {ride_name} and start frame {start_frame}.")
        return False

    # ---- 2) load all the data ----
    video_path   = ride_dir / f"seq_{start_frame}" / f"front_camera.mp4"
    path_tracker_path = ride_dir / f"seq_{start_frame}" / "path_tracker.h5"
    if not video_path.exists() or not path_tracker_path.exists():
        print(f"Video file {video_path} or {path_tracker_path} does not exist. Skipping ride {ride_name}.")
        return False

    path_info = hkl.load(path_tracker_path)
    path_mask = path_info["path_mask"][()] > 0  # (H,W) bool

    # decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(str(video_path))
    video_np = vr.get_batch(range(len(vr))).asnumpy()  # Load every frame from the video
    last_video_np = vr.get_batch(range(len(vr) - 1, len(vr))).asnumpy()  # technically earliest frame
    
    # ---- 3) run models ----
    image_predictor = sam2_model_dict['image_predictor']
    grounding_model = dino_model_dict['grounding_model']
    dino_processor = dino_model_dict['processor']
    video_predictor = sam2_model_dict['video_predictor']

    # Understand relevant objects in the scene
    try:
        # Blend the last frame with the path mask
        blend_color = (51, 255, 255)
        blend_frame_np = blend_mask(last_video_np[0], path_mask, color=blend_color, alpha=0.5)
        img_pil = Image.fromarray(blend_frame_np).convert("RGB")  # Convert to PIL Image
        
        if caption_only or PROCESS_CAPTION_ONLY:
            gt_caption_label = split_row["label"]
            seq_dir = (Path(out_dir) / f"ride_{driveid0}_{driveid1}_{timestamp}" /
               f"seq_{start_frame}")
            seq_dir.mkdir(parents=True, exist_ok=True)
            h5_path = seq_dir / "entity_caption.h5"

            # Look up the full_raw.txt and row to get the caption
            outputs = {}
            if h5_path.exists() and cache_enabled:
                outputs = hkl.load(str(h5_path))
            
            if not 'gt_scene_caption' in outputs or 'scene_caption' not in outputs:  
                scene_caption_list = process_image(gt_caption_label.strip(), img_pil, vlm_model_dict)

                outputs["gt_scene_caption"] = gt_caption_label.strip()
                outputs["scene_caption"] = scene_caption_list
                logging.info(f"Creating new caption file {h5_path}")
                hkl.dump(outputs, str(h5_path), mode="w")
                return True
            
            logging.info(f"Caption file {h5_path} already exists. Skipping caption processing.")
            return True

        assert image_predictor is not None, "Image predictor must be initialized"
        assert grounding_model is not None, "Grounding model must be initialized"
        assert dino_processor is not None, "DINO processor must be initialized"
        assert not PERFORM_TRACKING or video_predictor is not None, "Video predictor must be initialized"

        object_caption = process_image(QWEN_CAPTIONING_PROMPT, img_pil, vlm_model_dict)
        # scene_caption = process_image(QWEN_CAPTIONING_PROMPT, img_pil, vlm_processor, vlm_multi_model)
        if object_caption.strip() == "" or len(object_caption.strip()) == 0:
            print(f"No objects detected in the image for ride {ride_name}.")
            return False

        # Detect and track masks
        detections = segment_masks(
            image_np=last_video_np[0],
            prompt=object_caption,
            image_predictor=image_predictor,
            grounding_model=grounding_model,
            processor=dino_processor,
            device=device
        )
        if detections is None:
            # Image.fromarray(last_video_np[0]).save("test.jpg")
            print(f"No detections found for ride {ride_name}.")
            return False
    except Exception as e:
        print(f"Error processing ride {ride_name}: {e}")
        return False

    # ------------------------------------------------------------------ #
    #  3. validate masks                                                 #
    # ------------------------------------------------------------------ #
    # List of unwanted classes
    unwanted_classes = [
        "handlebars", "sky", "cloud", "grass", "ground", "road", "street", "sidewalk",
        "ground", "background", "terrain", "building", "wall", "floor", "ceiling",
        "concrete", "asphalt", "dirt", "sand", "rock", "path", "shadow", "brick", "bricks", "", 
        "purple", "purple light"
    ]

    if PERFORM_TRACKING:
        video_segments = track_masks(
            video_np=video_np,
            objects=detections["labels"],
            masks=detections["masks"],
            video_path=str(video_path),
            video_predictor=video_predictor
        )
        if video_segments is None:
            print(f"No video segments found for ride {ride_name}.")
            return False
        
        src_frame = max(video_segments.keys())
        if src_frame != len(video_np) - 1:
            print(f"Error: could not find segments for last frame {src_frame} "
                f"in video of ride {ride_name}.")
            return False

        final_masks   = video_segments[src_frame]        # dict {id: mask}
        labels        = np.array(detections["labels"])   # convenience
        valid_flags   = np.zeros(len(labels), dtype=bool)

        H, W = video_np[src_frame].shape[:2]  # height and width of the image
        for obj_id, mask in final_masks.items():
            idx = obj_id - 1                             # obj ids start at 1
            label = labels[idx].lower().strip()
            if mask.sum() == 0 or (mask.sum() / (H*W)) > 0.1:
                continue                                 # empty mask → invalid
            
            if label in unwanted_classes:
                print(f"Skipping unwanted class '{label}' for ride {ride_name}.")
                continue                                 # unwanted class → invalid

            valid_flags[idx] = True

        if valid_flags.sum() == 0:
            print(f"Error: no valid masks for ride {ride_name}")
            return False

        valid_ids      = np.nonzero(valid_flags)[0] + 1          # back to 1-based
        valid_segments = {i: m for i, m in final_masks.items() if i in valid_ids}
    else:
        # Filter object captions based on unwanted classes
        labels = np.array(detections["labels"])
        valid_flags   = np.zeros(len(labels), dtype=bool)
        for label in labels:
            if label.lower() in unwanted_classes:
                print(f"Skipping unwanted class '{label}' for ride {ride_name}.")
                continue
            valid_flags[labels == label] = True  # mark all instances of this label as valid
        if valid_flags.sum() == 0:
            print(f"Error: no valid masks for ride {ride_name}")
            return False
        
    # Updated detections valid
    detections_valid = {
        "masks":  detections["masks"][valid_flags],
        "scores": detections["scores"][valid_flags],
        # "logits": detections["logits"][valid_flags],
        "boxes":  detections["boxes"][valid_flags],
        "labels": labels[valid_flags],              # list[str] for HDF5
    }

    # Perform shape invariance ops
    T, H, W = detections_valid["masks"].shape[-3:]
    expected_shapes = {
        "masks": (T, H, W),
        "scores": (T, ),
        "boxes": (T, 4),
        "labels": (T, )
    }
    for key in expected_shapes.keys(): 
        if detections_valid[key].shape != expected_shapes[key]:
            logging.warning(f"Shape mismatch for {key}: expected {expected_shapes[key]}, got {detections_valid[key].shape}")
            return False  # Early exit if shapes do not match
        
    # ------------------------------------------------------------------ #
    #  4. keep only valid objects                                        #
    # ------------------------------------------------------------------ #
    detected_object_captions = ", ".join(detections_valid["labels"])

    if OBJECT_SELECTION_METHOD == "vlm":
        refine_prompt = f"""
        {detected_object_captions.strip()}
        Select the nearest object from the list above that the robot is navigating to in the format <action> <preposition> <object>. 
        Output ten descriptions with the same <object> and different <action> and <prepositions> tokens in a comma separated list format.
        If there is not a nearby or clear object, output an empty string.
        """
        best_key = None
    elif OBJECT_SELECTION_METHOD == "distance":
        video_segments = {
            i: m[None,...].astype(bool) for i, m in enumerate(detections_valid["masks"])
        }
        best_key, best_score = get_closest_entity_mask(
            video_segments=video_segments,
            path_mask=path_mask
        )
        class_name = detections_valid["labels"][best_key]
        refine_prompt = f"""
        Describe how the robot is navigating to the {class_name} <object> in the format <action> <preposition> <object>. 
        Output ten descriptions with the same <object> and different <action> and <prepositions> tokens in a comma separated list format.
        If there is not a nearby or clear object, output an empty string.
        """
        detections_valid["best_key"] = best_key
        detections_valid["best_score"] = best_score

    try:
        if best_key is not None:
            start_key, end_key = best_key, best_key + 1
        else:
            start_key, end_key = 0, len(detections_valid["masks"])
        img = draw_masks_single(
            image=last_video_np[0],
            masks=detections_valid["masks"][start_key:end_key],
            boxes=detections_valid["boxes"][start_key:end_key],
            labels=detections_valid["labels"][start_key:end_key]
        )
    except Exception as e:
        print(f"Error drawing masks for ride {ride_name}: {e}")
        img = last_video_np[0]  # Fallback to the original image if drawing fails

    if detections_valid["masks"].ndim == 4:
        detections_valid["masks"] = detections_valid["masks"].squeeze(1)

    img = Image.fromarray(img).convert("RGB") 
    refined_object_caption = process_image(refine_prompt, img, vlm_model_dict)

    refined_object_captions_list = refined_object_caption.strip().split(",")
    if len(refined_object_captions_list) <= 2:
        print(f"Error: no valid objects found in the image for ride {ride_name}.")
        return False

    if save_visualizations:
        annotated = None
        if PERFORM_TRACKING:
            # ---- keep only masks that survived validation -----------------
            segments    = valid_segments            # dict {obj_id : (1,H,W) mask}
            object_ids  = sorted(segments.keys())   # deterministic order
            masks       = np.concatenate([segments[i] for i in object_ids], axis=0)

            # image of the last frame
            img = video_np[src_frame]

            # map every original id → label string  (labels already filtered)
            id_to_label = {obj_id: label
                        for obj_id, label in zip(valid_ids, detections_valid["labels"])}

            # ---- Supervision ---------------------------------------------------
            sv_detections = sv.Detections(
                xyxy     = sv.mask_to_xyxy(masks),               # (n,4)
                mask     = masks,                                # (n,H,W)
                class_id = np.array(object_ids, dtype=np.int32), # (n,)
            )

            # round box
            box_annotator = sv.RoundBoxAnnotator()
            annotated     = box_annotator.annotate(
                scene=img.copy(),
                detections=sv_detections
            )

            # label underneath the box
            label_annotator = sv.LabelAnnotator(
                text_position=sv.Position.BOTTOM_CENTER
            )
            annotated = label_annotator.annotate(
                annotated,
                detections=sv_detections,
                labels=[id_to_label[i] for i in object_ids],
            )

            # mask overlay
            mask_annotator = sv.MaskAnnotator()
            annotated = mask_annotator.annotate(
                scene=annotated,
                detections=sv_detections
            )
        else:
            # ---- draw masks on the earliest temporal frame -----------------------------
            annotated = draw_masks_single(
                image=last_video_np[0],
                masks=detections_valid["masks"],
                boxes=detections_valid["boxes"],
                labels=detections_valid["labels"]
            )
            # Blend the path mask on the last frame
            blend_color = (51, 255, 255)
            annotated = blend_mask(annotated, path_mask, color=blend_color, alpha=0.5)
            # Concatenated last and first frame
            annotated = np.concatenate([annotated, video_np[0]], axis=1)
            # Write the closest name on the image
            cv2.putText(
                annotated,
                f"Closest object: {class_name} Distance {best_score:.2f}px",
                (10, 30),  # Position at the top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # ---- write to disk -------------------------------------------------
        vis_dir  = Path("entity_masks_outputs")
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / f"ride_{ride_id}_{driveid0}_{driveid1}_{timestamp}_{start_frame}.jpg"
        Image.fromarray(annotated).save(vis_path)
        Image.fromarray(annotated).save("test.jpg")

    # ------------------------------------------------------------------ #
    #  6. save                                                           #
    # ------------------------------------------------------------------ #
    seq_dir = (Path(out_dir) / f"ride_{driveid0}_{driveid1}_{timestamp}" /
               f"seq_{start_frame}")
    seq_dir.mkdir(parents=True, exist_ok=True)
    h5_path = seq_dir / "entity_info.h5"

    outputs = {
        **detections_valid,             # masks / boxes / scores / logits / labels
        "refined_object_caption": refined_object_caption,
        "object_caption": object_caption,
    }
    if PERFORM_TRACKING:
        outputs["video_segments"] = video_segments
    hkl.dump(outputs, str(h5_path), mode="w")
    return True









