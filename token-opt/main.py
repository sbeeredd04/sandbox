import os
import yaml
import torch
import tqdm
from PIL import Image 
import torchvision.transforms as T
from pathlib import Path
from typing import Dict, Any

from tto.test_time_opt import TestTimeOptConfig, TestTimeOpt, CLIPObjective, TestTimeOptInfo


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Configs : {config}")
    return config


def setup_device(gpu: str) -> torch.device:
    """Setup device to use GPU(s)."""
    print(f"Using GPU(s): {gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")
    return device


def load_and_preprocess_image(image_path: str, image_size: int, mean: list, std: list, device: torch.device) -> torch.Tensor:
    """Load and preprocess input image."""
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    image = transform(image).unsqueeze(0).to(device)
    print(f"Input image loaded and preprocessed: {image_path} --> tensor shape {image.shape}")
    return image


def save_image(tensor: torch.Tensor, output_path: str, mean: list, std: list):
    """Save tensor as image, denormalizing if needed."""
    # Denormalize
    tensor = tensor.squeeze(0).cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    
    result_image = T.ToPILImage()(tensor)
    result_image.save(output_path)
    print(f"Image saved to: {output_path}")


def image_editing(config: Dict[str, Any]): 

    print("=="*20, "Image Editing Task", "=="*20)
    print(f"Image editing: {config['input_path']} with prompt: {config['prompt']} --> {config['output_path']}")

    # Setup device
    device = setup_device(config['gpu'])
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Load and preprocess input image
    image = load_and_preprocess_image(
        config['input_path'],
        config['image']['size'],
        config['image']['normalize_mean'],
        config['image']['normalize_std'],
        device
    )
    
    # Initialize test-time optimization config
    opt_config = TestTimeOptConfig(
        titok_checkpoint=config['optimization']['titok_checkpoint'],
        optimize_post_quantization_tokens=config['optimization']['optimize_post_quantization_tokens'],
        vae_deterministic_sampling=config['optimization']['vae_deterministic_sampling'],
        lr=config['optimization']['lr'],
        num_iter=config['optimization']['num_iter'],
        ema_decay=config['optimization']['ema_decay'],
        token_noise=config['optimization']['token_noise'],
        reg_weight=config['optimization']['reg_weight'],
        reg_type=config['optimization']['reg_type'],
        enable_amp=config['optimization']['enable_amp'],
    )
    
    # Initialize CLIP objective
    clip_objective = CLIPObjective(
        prompt=config['prompt'],
        neg_prompt=config.get('neg_prompt'),
        cfg_scale=config['clip']['cfg_scale'],
        num_augmentations=config['clip']['num_augmentations'],
        pretrained=(config['clip']['pretrained_model'], config['clip']['pretrained_checkpoint'])
    ).to(device)
    
    # Initialize optimizer
    optimizer = TestTimeOpt(config=opt_config, objective=clip_objective).to(device)
    
    # Progress callback
    log_interval = config.get('log_interval', 50)
    def progress_callback(info: TestTimeOptInfo): 
        if info.i % log_interval == 0: 
            loss_val = info.loss.mean().item() if info.loss.numel() > 1 else info.loss.item()
            print(f"Iteration {info.i}/{opt_config.num_iter}, Loss: {loss_val:.4f}")
        return False  # Don't stop early
    
    # Run optimization
    print("Starting test-time optimization...")
    optimized_image = optimizer(
        seed=image,
        callback=progress_callback,
    )
        
    # Save optimized image
    save_image(
        optimized_image,
        config['output_path'],
        config['image']['normalize_mean'],
        config['image']['normalize_std']
    )
    print("Image editing completed successfully!")


def generate_image(config: Dict[str, Any]): 

    print("=="*20, "Image Generation Task", "=="*20)
    print(f"Generating image with prompt: {config['prompt']} --> {config['output_path']}")
    
    # Setup device
    device = setup_device(config['gpu'])
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Initialize test-time optimization config
    opt_config = TestTimeOptConfig(
        titok_checkpoint=config['optimization']['titok_checkpoint'],
        optimize_post_quantization_tokens=config['optimization']['optimize_post_quantization_tokens'],
        vae_deterministic_sampling=config['optimization']['vae_deterministic_sampling'],
        lr=config['optimization']['lr'],
        num_iter=config['optimization']['num_iter'],
        ema_decay=config['optimization']['ema_decay'],
        token_noise=config['optimization']['token_noise'],
        reg_weight=config['optimization']['reg_weight'],
        reg_type=config['optimization']['reg_type'],
        enable_amp=config['optimization']['enable_amp'],
    )
    
    # Initialize CLIP objective
    clip_objective = CLIPObjective(
        prompt=config['prompt'],
        neg_prompt=config.get('neg_prompt'),
        cfg_scale=config['clip']['cfg_scale'],
        num_augmentations=config['clip']['num_augmentations'],
        pretrained=(config['clip']['pretrained_model'], config['clip']['pretrained_checkpoint'])
    ).to(device)
    
    # Initialize optimizer
    optimizer = TestTimeOpt(config=opt_config, objective=clip_objective).to(device)
    
    # Initialize random tokens for generation
    # Get token dimensions from the model
    token_dim = optimizer.titok.encoder.token_size  # Channel dimension (e.g., 12)
    num_tokens = optimizer.titok.num_latent_tokens  # Number of latent tokens (e.g., 32)
    
    # Create random seed tokens
    batch_size = 1
    seed_tokens = torch.randn(batch_size, token_dim, 1, num_tokens, device=device)
    
    # Progress callback
    log_interval = config.get('log_interval', 50)
    def progress_callback(info: TestTimeOptInfo): 
        if info.i % log_interval == 0: 
            loss_val = info.loss.mean().item() if info.loss.numel() > 1 else info.loss.item()
            print(f"Iteration {info.i}/{opt_config.num_iter}, Loss: {loss_val:.4f}")
        return False
    
    # Run optimization from random tokens
    print("Starting test-time optimization from random tokens...")
    generated_image = optimizer(
        seed=None,
        seed_tokens=seed_tokens,
        callback=progress_callback,
    )
        
    # Save generated image
    save_image(
        generated_image,
        config['output_path'],
        config['image']['normalize_mean'],
        config['image']['normalize_std']
    )
    print("Image generation completed successfully!")


def image_inpainting(config: Dict[str, Any]):
    """Perform image inpainting using test-time optimization."""
    print("=="*20, "Image Inpainting Task", "=="*20)
    print("Note: Image inpainting requires a mask. This is a placeholder implementation.")
    print(f"Inpainting: {config['input_path']} with prompt: {config['prompt']} --> {config['output_path']}")
    
    # Setup device
    device = setup_device(config['gpu'])
    
    # Set random seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Load input image
    image = load_and_preprocess_image(
        config['input_path'],
        config['image']['size'],
        config['image']['normalize_mean'],
        config['image']['normalize_std'],
        device
    )
    
    # TODO: Load mask (binary mask indicating regions to inpaint)
    # 1. Load a mask image
    # 2. Create a custom objective that only optimizes masked regions
    # 3. Use the mask during optimization to blend original and generated content
    
    print("Warning: Inpainting not fully implemented. Falling back to image editing.")
    
    # Use image editing as fallback
    image_editing(config)


def main(config_path: str): 
    """Main entry point for test-time optimization."""
    # Load configuration from YAML
    config = load_config(config_path)
    
    # Validate required fields
    if 'task' not in config:
        raise ValueError("Configuration must specify 'task' field")
    
    task = config['task']
    
    # Create output directory if it doesn't exist
    output_dir = Path(config['output_path']).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute the appropriate task
    if task == 'image_editing':
        if 'input_path' not in config or not config['input_path']:
            raise ValueError("Image editing requires 'input_path' in config")
        if 'prompt' not in config or not config['prompt']:
            raise ValueError("Image editing requires 'prompt' in config")
        image_editing(config)
    
    elif task == 'generate_image':
        if 'prompt' not in config or not config['prompt']:
            raise ValueError("Image generation requires 'prompt' in config")
        generate_image(config)

    elif task == 'image_inpainting':
        if 'input_path' not in config or not config['input_path']:
            raise ValueError("Image inpainting requires 'input_path' in config")
        if 'prompt' not in config or not config['prompt']:
            raise ValueError("Image inpainting requires 'prompt' in config")
        image_inpainting(config)
    
    else:
        raise ValueError(f"Unknown task: {task}. Must be one of: image_editing, generate_image, image_inpainting")





if __name__ == "__main__": 
    import argparse
    
    # Simple argument parser for config file path
    parser = argparse.ArgumentParser(description="Test-time optimization for image processing")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    # Run main with config file
    main(config_path=args.config)
    
    # Example usage:
    # python token-opt/main.py --config token-opt/configs/config.yaml
