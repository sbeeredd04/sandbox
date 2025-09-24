from __future__ import print_function

import argparse
import os
import time
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
import glob
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

from model import ResNet18
from utils import Logger, mkdir_p, savefig
from train_utils import (
    WarmupCosineSchedule, save_checkpoint, train, test, 
    train_olympic_action, test_olympic_action, 
    train_ucf_sports, test_ucf_sports
)
# UCF imports are conditional to avoid dependency issues
try:
    from ucf_action_utils import UCFSportsDataset, get_ucf_sports_transforms
except ImportError:
    print("Warning: UCF Sports utilities not available due to missing dependencies")
# DeepLake import is conditional for UCF Sports
try:
    import deeplake
    ds = deeplake.load('hub://activeloop/ucf-sports-action')
except ImportError:
    ds = None
    print("Warning: DeepLake not available for UCF Sports")

wandb.login()

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N', help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=100, type=int, metavar='N', help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

# Options to run inference on a model checkpoint path
parser.add_argument('--inference', default='', type=str, metavar='PATH', help='path to model checkpoint (default: none)')
parser.add_argument('--model-path', default='', type=str, metavar='PATH', help='path to model checkpoint (default: none)')
parser.add_argument('--image-path', default='', type=str, metavar='PATH', help='path to image (default: none)')

# Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA with proper multi-GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def run_inference(model_path, image_path, use_cuda=True):
    
    # Determine if we're processing a single image or multiple images
    if os.path.isdir(image_path):
        # Get all image files from the directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_path, ext)))
            image_files.extend(glob.glob(os.path.join(image_path, ext.upper())))
        
        if not image_files:
            print(f"No image files found in directory: {image_path}")
            return []
        
        print(f"Found {len(image_files)} image files in directory: {image_path}")
        is_batch_processing = True
    else:
        # Single image file
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
        image_files = [image_path]
        is_batch_processing = False
    
    # Get UCF Sports class names using utility functions with grouping
    from ucf_action_utils import get_ucf_class_mappings
    
    (grouped_class_names, original_to_grouped_id, grouped_to_original_ids, 
     get_class_name, get_grouped_class_id, grouping_rules) = get_ucf_class_mappings(ds, group_similar=True)
    
    print(f"Grouped class names: {grouped_class_names}")
    print(f"Original to grouped ID mapping: {original_to_grouped_id}")
    
    # Load the model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model architecture - use grouped classes count
    from model import DGMResNet, BasicBlock
    model = DGMResNet(BasicBlock, num_classes=len(grouped_class_names), hw=256)
    print(f"Created inference model with {len(grouped_class_names)} output classes (grouped)")
    
    # Load the state dict - handle DataParallel prefix
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix from keys if present (DataParallel saves with this prefix)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    
    # Move to GPU if available
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu' 
    
    model.eval()
    
    # Use the same transforms as training (without augmentation)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process all images
    results = []
    all_predictions = []
    
    for i, img_file in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(img_file)}")
        
        try:
            # Load image
            image = Image.open(img_file).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            
            if use_cuda and torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            # Run inference
            with torch.no_grad():
                outputs, imgr1, imgr2, imgr3, imgr4 = model(input_tensor)
                
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                predicted_class = predicted_class.item()
                confidence = confidence.item()
                # Model now outputs grouped class IDs directly
                class_name = grouped_class_names[predicted_class] if predicted_class < len(grouped_class_names) else "Unknown"
                
                # Store results
                result = {
                    'filename': os.path.basename(img_file),
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'class_name': class_name,
                    'probabilities': probabilities[0].cpu().numpy()
                }
                results.append(result)
                all_predictions.append(predicted_class)
                
                # Create visualization for this image
                plt.figure(figsize=(15, 5))
                
                # Plot 1: Original image (denormalized)
                plt.subplot(1, 3, 1)
                # Denormalize the input tensor for display
                mean_norm = np.array([0.485, 0.456, 0.406])
                std_norm = np.array([0.229, 0.224, 0.225])
                denorm_img = input_tensor[0].cpu().numpy()
                denorm_img = (denorm_img * std_norm[:, None, None]) + mean_norm[:, None, None]
                denorm_img = np.clip(denorm_img, 0, 1)
                denorm_img = np.transpose(denorm_img, (1, 2, 0))  # C,H,W to H,W,C
                plt.imshow(denorm_img)
                plt.title(f'Original Image\n{os.path.basename(img_file)}')
                plt.axis('off')
                
                # Plot 2: IMRG visualization
                plt.subplot(1, 3, 2)
                plt.imshow(imgr4[0, 0].cpu().numpy(), cmap='viridis')
                plt.title(f'IMRG Visualization\nPredicted: {class_name}')
                plt.axis('off')
                
                # Plot 3: Class probabilities bar chart (already grouped)
                plt.subplot(1, 3, 3)
                probs = probabilities[0].cpu().numpy()
                
                # Model now outputs grouped class probabilities directly
                bars = plt.bar(range(len(grouped_class_names)), probs)
                bars[predicted_class].set_color('red')  # Highlight predicted class
                plt.title(f'Class Probabilities\nConfidence: {confidence:.3f}')
                plt.xlabel('Class Index')
                plt.ylabel('Probability')
                plt.xticks(range(len(grouped_class_names)), [f'{i}' for i in range(len(grouped_class_names))], rotation=45)
                
                plt.tight_layout()
                
                # Save individual result
                safe_filename = os.path.splitext(os.path.basename(img_file))[0]
                plt.savefig(f'./data/debug/inference_{safe_filename}_{class_name}.png', dpi=150, bbox_inches='tight')
                
                # Log individual image to wandb
                wandb.log({
                    f'inference_{safe_filename}': wandb.Image(plt.gcf())
                })
                plt.close()
                
                print(f"  -> Predicted: {class_name} (confidence: {confidence:.4f})")
                
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue
    
    # Create summary visualization if processing multiple images
    if is_batch_processing and results:
        create_batch_summary(results, grouped_class_names, image_path)
    
    # Return results
    if is_batch_processing:
        return results
    else:
        if results:
            result = results[0]
            return result['predicted_class'], result['confidence'], result['class_name']
        else:
            return None, None, None


def create_batch_summary(results, class_names, image_path):
    """
    Create a summary visualization for batch inference results
    """
    
    # Count predictions
    predictions_count = Counter([r['class_name'] for r in results])
    confidences = [r['confidence'] for r in results]
    
    # Create summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Prediction counts
    classes = list(predictions_count.keys())
    counts = list(predictions_count.values())
    bars1 = ax1.bar(classes, counts)
    ax1.set_title(f'Prediction Distribution\n({len(results)} images processed)')
    ax1.set_xlabel('Predicted Class')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                str(count), ha='center', va='bottom')
    
    # Plot 2: Confidence distribution
    ax2.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_title('Confidence Score Distribution')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(confidences), color='red', linestyle='--', 
                label=f'Mean: {np.mean(confidences):.3f}')
    ax2.legend()
    
    # Plot 3: Top predictions with confidence
    top_results = sorted(results, key=lambda x: x['confidence'], reverse=True)[:10]
    filenames = [r['filename'][:15] + '...' if len(r['filename']) > 15 else r['filename'] 
                for r in top_results]
    top_confidences = [r['confidence'] for r in top_results]
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_results)))
    
    bars3 = ax3.barh(range(len(top_results)), top_confidences, color=colors)
    ax3.set_yticks(range(len(top_results)))
    ax3.set_yticklabels([f"{r['class_name']}" for r in top_results])
    ax3.set_xlabel('Confidence Score')
    ax3.set_title('Top 10 Most Confident Predictions')
    
    # Add confidence values on bars
    for i, (bar, conf) in enumerate(zip(bars3, top_confidences)):
        ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{conf:.3f}', ha='left', va='center', fontsize=8)
    
    # Plot 4: Average confidence per class
    class_confidences = {}
    for result in results:
        class_name = result['class_name']
        if class_name not in class_confidences:
            class_confidences[class_name] = []
        class_confidences[class_name].append(result['confidence'])
    
    avg_confidences = {k: np.mean(v) for k, v in class_confidences.items()}
    classes_avg = list(avg_confidences.keys())
    avg_conf_values = list(avg_confidences.values())
    
    bars4 = ax4.bar(classes_avg, avg_conf_values, alpha=0.7)
    ax4.set_title('Average Confidence by Class')
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Average Confidence')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add average values on bars
    for bar, avg_conf in zip(bars4, avg_conf_values):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{avg_conf:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save summary plot
    folder_name = os.path.basename(os.path.normpath(image_path))
    summary_path = f'./data/debug/batch_inference_summary_{folder_name}.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Batch summary saved to: {summary_path}")
    
    # Log summary to wandb
    wandb.log({
        'batch_inference_summary': wandb.Image(plt.gcf()),
        'total_images_processed': len(results),
        'unique_classes_predicted': len(predictions_count),
        'average_confidence': np.mean(confidences),
        'prediction_distribution': dict(predictions_count)
    })
    plt.close()
    
    # Print summary statistics
    print(f"\n=== BATCH INFERENCE SUMMARY ===")
    print(f"Total images processed: {len(results)}")
    print(f"Unique classes predicted: {len(predictions_count)}")
    print(f"Average confidence: {np.mean(confidences):.4f}")
    print(f"Confidence range: {min(confidences):.4f} - {max(confidences):.4f}")
    print(f"\nClass distribution:")
    for class_name, count in sorted(predictions_count.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        avg_conf = np.mean([r['confidence'] for r in results if r['class_name'] == class_name])
        print(f"  {class_name}: {count} images ({percentage:.1f}%) - avg conf: {avg_conf:.3f}")


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    # Handle inference mode
    if args.model_path and args.image_path:
        if os.path.isdir(args.image_path):
            print(f"Running batch inference on folder: {args.image_path}")
        else:
            print(f"Running inference on image: {args.image_path}")
            
        if not os.path.exists(args.image_path):
            print(f"Error: Path {args.image_path} not found!")
            return
        if not os.path.exists(args.model_path):
            print(f"Error: Model file {args.model_path} not found!")
            return
        
        results = run_inference(args.model_path, args.image_path, use_cuda)
        
        if isinstance(results, list):
            # Batch processing results
            print(f"\nBatch Inference Complete!")
            print(f"Processed {len(results)} images")
            if results:
                print(f"Results saved to ./data/debug/")
        elif results[0] is not None:
            # Single image results
            predicted_class, confidence, class_name = results
            print(f"\nInference Results:")
            print(f"Predicted Class: {predicted_class} ({class_name})")
            print(f"Confidence: {confidence:.4f}")
        else:
            print("Inference failed - no results obtained")
        return
    
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    transforms1 = transforms.RandomApply(torch.nn.ModuleList([transforms.RandomAffine(90, translate=(0.2, 0.2), scale = (0.6, 1.3)),]), p=0.4)
    transforms2 = transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(0.8, 0.8, 0.8, 0.25),]), p=0.3)
    transform_train = transforms.Compose([
        transforms2,
        transforms1,
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
        
    elif args.dataset == 'olympic_action':
        # Use custom dataset module
        num_classes = 16
        
    elif args.dataset == 'ucf_sports':
        # UCF Sports Action dataset
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100
        
    # Olympic Action dataset
    if args.dataset == 'olympic_action':
        print("Loading Olympic Action dataset...")
        # Import Olympic Action dataset utilities
        from utils.olympic_action import create_olympic_action_dataloaders
        
        # Create Olympic Action datasets and dataloaders
        train_loader, val_loader, num_classes = create_olympic_action_dataloaders(
            root_dir='./data/olympic_sports',
            batch_size=args.train_batch,
            num_workers=args.workers,
            num_frames_per_video=16
        )
        print(f"Olympic Action dataset loaded successfully with {num_classes} classes")
        
    elif args.dataset == 'ucf_sports':
        # UCF Sports Action dataset transforms
        transform = get_ucf_sports_transforms()
        
        # Create UCF Sports datasets with grouped classes
        trainset = UCFSportsDataset(ds, split='train', transform=transform, use_grouped_classes=True)
        train_loader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        
        # Test loader
        testset = UCFSportsDataset(ds, split='test', transform=transform, use_grouped_classes=True)
        val_loader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        
        # Update num_classes to use grouped classes count
        num_classes = trainset.get_num_classes()
        print(f"Using {num_classes} grouped classes for training")
        
    else:
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        train_loader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        
        # Test loader
        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        val_loader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Create model with appropriate image dimensions
    if args.dataset == 'ucf_sports':
        # UCF Sports uses 224x224 images
        from model import DGMResNet, BasicBlock
        model = DGMResNet(BasicBlock, num_classes=num_classes, hw=256)
        print(f"Created model with {num_classes} output classes")
    elif args.dataset == 'olympic_action':
        # Olympic Action uses 224x224 images (resized from 480x360)
        from model import DGMResNet, BasicBlock
        model = DGMResNet(BasicBlock, num_classes=num_classes, hw=256)
    else:
        # CIFAR uses 32x32 images
        model = ResNet18(num_classes=num_classes)
    
    # GPU setup with proper multi-GPU support for all datasets
    if use_cuda:
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel for {args.dataset}")
            
            # Clear GPU cache before DataParallel to avoid memory issues
            torch.cuda.empty_cache()
            
            # Initialize model on GPU first
            model = model.cuda()
            
            try:
                # Try DataParallel with NCCL backend fixes
                print("Attempting DataParallel initialization...")
                model = torch.nn.DataParallel(model)
                
                # Test a small forward pass to verify DataParallel works
                print("Testing DataParallel with dummy input...")
                test_input = torch.randn(2, 3, 256, 256).cuda()
                with torch.no_grad():
                    _ = model(test_input)
                print("DataParallel test successful!")
                
                print(f"DataParallel successfully initialized on GPUs: {list(range(torch.cuda.device_count()))}")
                
            except Exception as e:
                print(f"DataParallel failed: {e}")
                print("Falling back to single GPU...")
                
                # Extract model from DataParallel if it was wrapped
                if isinstance(model, torch.nn.DataParallel):
                    model = model.module
                
                # Clear cache and use single GPU
                torch.cuda.empty_cache()
                model = model.cuda()
                print("Successfully switched to single GPU mode")
        else:
            print(f"Using single GPU for {args.dataset} training")
            model = model.cuda()
    else:
        print("Using CPU for training")

    cudnn.benchmark = True
    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=100, t_total=args.epochs*len(train_loader))

    # Resume
    if args.dataset == 'olympic_action':
        title = 'olympic_action-DGM-ResNet18'
    elif args.dataset == 'ucf_sports':
        title = 'ucf-sports-DGM-ResNet18'
    else:
        title = 'cifar-DGM-ResNet18'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate: 
        print('\nEvaluation only')
        if args.dataset == 'ucf_sports':
            test_loss, test_acc = test_ucf_sports(val_loader, model, criterion, start_epoch, use_cuda)
        elif args.dataset == 'olympic_action':
            test_loss, test_acc = test_olympic_action(val_loader, model, criterion, start_epoch, use_cuda)
        else:
            test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, scheduler.get_last_lr()[0]))
        
        try:
            if args.dataset == 'olympic_action':
                train_loss, train_acc = train_olympic_action(train_loader, model, criterion, optimizer, epoch, use_cuda, scheduler)
                test_loss, test_acc = test_olympic_action(val_loader, model, criterion, epoch, use_cuda)
            elif args.dataset == 'ucf_sports':
                train_loss, train_acc = train_ucf_sports(train_loader, model, criterion, optimizer, epoch, use_cuda, scheduler)
                test_loss, test_acc = test_ucf_sports(val_loader, model, criterion, epoch, use_cuda)
            else:
                train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda, scheduler)
                test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)
        except RuntimeError as e:
            error_msg = str(e)
            raise e

        # Append logger file
        logger.append([scheduler.get_last_lr()[0], train_loss, test_loss, train_acc, test_acc])

        # Save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)
        
        # Log to wandb
        run.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
        })
        
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


if __name__ == '__main__':
    # Get the arg of the current dataset 
    dataset_arg = args.dataset
    
    if dataset_arg == 'ucf_sports':
        project_name = "ucf-sports-dgm-resnet18"
    elif dataset_arg == 'olympic_action':
        project_name = "olympic-action-dgm-resnet18"
    else:
        project_name = "cifar-dgm-resnet18"

    with wandb.init(project=project_name, name=project_name) as run:
        wandb.config.update(args)
        main()