
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np
from PIL import Image
from model import ResNet18
from torch.optim.lr_scheduler import LambdaLR
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import math
import wandb
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

#Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
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

class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

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
        #use custom dataset module
        num_classes = 16
        
    elif args.dataset == 'ucf_sports':
        # UCF Sports Action dataset
        num_classes = 13 
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
        transform_train, transform_test = get_ucf_sports_transforms()
        
        # Load UCF Sports dataset using Deep Lake
        import deeplake
        ds = deeplake.load('hub://activeloop/ucf-sports-action')
        
        # Create UCF Sports datasets
        trainset = UCFSportsDataset(ds, split='train', transform=transform_train)
        train_loader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        
        # Test loader
        testset = UCFSportsDataset(ds, split='test', transform=transform_test)
        val_loader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        
    else:
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        train_loader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        
        #test loader
        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        val_loader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Create model with appropriate image dimensions
    if args.dataset == 'ucf_sports':
        # UCF Sports uses 224x224 images
        from model import DGMResNet, BasicBlock
        model = DGMResNet(BasicBlock, num_classes=num_classes, hw=224)
    elif args.dataset == 'olympic_action':
        # Olympic Action uses 224x224 images (resized from 480x360)
        from model import DGMResNet, BasicBlock
        model = DGMResNet(BasicBlock, num_classes=num_classes, hw=224)
    else:
        # CIFAR uses 32x32 images
        model = ResNet18(num_classes=num_classes)
    
    model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    # define loss function (criterion) and optimizer
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
        #start_epoch = checkpoint['epoch']
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
        
        if args.dataset == 'olympic_action':
            print(f"Training Olympic Action model for epoch {epoch + 1}")
            train_loss, train_acc = train_olympic_action(train_loader, model, criterion, optimizer, epoch, use_cuda, scheduler)
            print(f"Testing Olympic Action model for epoch {epoch + 1}")
            test_loss, test_acc = test_olympic_action(val_loader, model, criterion, epoch, use_cuda)
        elif args.dataset == 'ucf_sports':
            train_loss, train_acc = train_ucf_sports(train_loader, model, criterion, optimizer, epoch, use_cuda, scheduler)
            test_loss, test_acc = test_ucf_sports(val_loader, model, criterion, epoch, use_cuda)

        else:
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda, scheduler)
            test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([scheduler.get_last_lr()[0], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)
        
        #log to wandb
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

def train(train_loader, model, criterion, optimizer, epoch, use_cuda, scheduler):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs, imgr = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs, imgr = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def train_olympic_action(train_loader, model, criterion, optimizer, epoch, use_cuda, scheduler):
    
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    end = time.time()
    bar = Bar('Processing', max=len(train_loader))
    
    print(f"Starting Olympic Action training for epoch {epoch + 1}")
    print(f"Number of training batches: {len(train_loader)}")
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        # Debug: Print input shape for first batch
        if batch_idx == 0:
            print(f"Input shape: {inputs.shape}")
            print(f"Targets shape: {targets.shape}")
            print(f"Targets: {targets}")
        
        # Handle video data: inputs shape is (batch_size, num_frames, C, H, W)
        # Take a random frame from each video sequence for training
        batch_size, num_frames, C, H, W = inputs.shape
        frame_indices = torch.randint(0, num_frames, (batch_size,))
        inputs = inputs[torch.arange(batch_size), frame_indices]  # Shape: (batch_size, C, H, W)
        
        if batch_idx == 0:
            print(f"After frame selection - Input shape: {inputs.shape}")
        
        # compute output
        outputs, imgr = model(inputs)
        loss = criterion(outputs, targets)
        
        if batch_idx == 0:
            print(f"Output shape: {outputs.shape}")
            print(f"Loss: {loss.item()}")
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    print(f"Olympic Action training completed for epoch {epoch + 1}")
    print(f"Final training loss: {losses.avg:.4f}, Final training accuracy: {top1.avg:.4f}")
    return (losses.avg, top1.avg)

def test_olympic_action(val_loader, model, criterion, epoch, use_cuda):
    global best_acc
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()
    
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    
    print(f"Starting Olympic Action testing for epoch {epoch + 1}")
    print(f"Number of test batches: {len(val_loader)}")
    
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        
        # Debug: Print input shape for first batch
        if batch_idx == 0:
            print(f"Test input shape: {inputs.shape}")
            print(f"Test targets shape: {targets.shape}")
            print(f"Test targets: {targets}")
        
        # Handle video data: inputs shape is (batch_size, num_frames, C, H, W)
        # For testing, take the middle frame from each video sequence
        batch_size, num_frames, C, H, W = inputs.shape
        middle_frame = num_frames // 2
        inputs = inputs[:, middle_frame]  # Shape: (batch_size, C, H, W)
        
        if batch_idx == 0:
            print(f"After frame selection - Test input shape: {inputs.shape}")
        
        # compute output
        outputs, imgr = model(inputs)
        loss = criterion(outputs, targets)
        
        if batch_idx == 0:
            print(f"Test output shape: {outputs.shape}")
            print(f"Test loss: {loss.item()}")
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    print(f"Olympic Action testing completed for epoch {epoch + 1}")
    print(f"Final test loss: {losses.avg:.4f}, Final test accuracy: {top1.avg:.4f}")
    return (losses.avg, top1.avg)

def get_ucf_sports_transforms():
    """Get UCF Sports Action dataset transforms with appropriate scaling"""
    # UCF Sports images are 720x480, we'll resize to 224x224 for ResNet18
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256
        transforms.RandomCrop(224),     # Random crop to 224x224
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    return transform_train, transform_test

class UCFSportsDataset(data.Dataset):
    """UCF Sports Action Dataset using Deep Lake"""
    def __init__(self, deeplake_ds, split='train', transform=None):
        self.deeplake_ds = deeplake_ds
        self.transform = transform
        self.split = split
        
        # Create stratified train/test split
        total_samples = len(deeplake_ds)
        
        # Get all labels first to create stratified split
        all_labels = []
        for i in range(total_samples):
            sample = deeplake_ds[i]
            label = int(sample.labels.numpy()[0])
            all_labels.append(label)
        
        # Create stratified indices
        from collections import defaultdict
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(all_labels):
            label_to_indices[label].append(idx)
        
        # Split each class 80/20
        train_indices = []
        test_indices = []
        
        for label, indices in label_to_indices.items():
            n_train = int(0.8 * len(indices))
            train_indices.extend(indices[:n_train])
            test_indices.extend(indices[n_train:])
        
        # Shuffle the indices
        import random
        random.shuffle(train_indices)
        random.shuffle(test_indices)
        
        if split == 'train':
            self.indices = train_indices
        else:  # test
            self.indices = test_indices
        
        run.log({
            'split': split,
            'num_samples': len(self.indices),
            'label_distribution': self._get_label_distribution(),
        })
    
    def _get_label_distribution(self):
        """Get distribution of labels in current split"""
        label_counts = {}
        for i, sample in enumerate(self.deeplake_ds):
            #for all samples
            label = int(sample.labels.numpy()[0])
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index from our split indices
        actual_idx = self.indices[idx]
        
        # Get image and label from Deep Lake dataset
        sample = self.deeplake_ds[actual_idx]        
        
        # Extract image and label
        image = sample.images.numpy()
        label = int(sample.labels.numpy()[0])
        
        # Store original image for return (create a processed version for wandb logging)
        original_image_for_wandb = self._prepare_image_for_wandb(image.copy())
        
        # Handle different image formats from Deep Lake
        if len(image.shape) == 4:  # (1, C, H, W) format
            image = image.squeeze(0)
        elif len(image.shape) == 3:  # (C, H, W) format
            pass
        else:
            image = image.squeeze()
        
        # Ensure we have a 3D tensor (C, H, W)
        if len(image.shape) != 3:
            raise ValueError(f"Unexpected image shape after processing: {image.shape}")
        
        # Convert CHW to HWC for PIL
        image = image.transpose(1, 2, 0)
        
        # Handle different data types
        if image.dtype == np.uint8:
            pass
        elif image.dtype in [np.float32, np.float64]:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Ensure values are in valid range
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        try:
            image = Image.fromarray(image, mode='RGB')
        except Exception as e:
            print(f"Error converting image to PIL: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Resize original image to consistent size for batching
        original_pil_resized = image.resize((224, 224))  # Resize to consistent size
        original_pil_array = np.array(original_pil_resized)
        
        # Apply transforms
        transformed_image = None
        if self.transform:
            transformed_image = self.transform(image)
        else:
            # Convert to tensor if no transforms
            transform_to_tensor = transforms.ToTensor()
            transformed_image = transform_to_tensor(image)
        
        # Return: (transformed_tensor, label, original_as_numpy, original_for_wandb)
        return transformed_image, label, original_pil_array, original_image_for_wandb
    
    def _prepare_image_for_wandb(self, np_image):
        """Prepare numpy image for wandb logging"""
        # Create a copy to avoid modifying original
        img_for_log = np_image.copy()
        
        # Handle different image formats from Deep Lake
        if len(img_for_log.shape) == 4:  # (1, C, H, W) format
            img_for_log = img_for_log.squeeze(0)
        
        # If image is in (C, H, W) format, convert to (H, W, C) for wandb
        if len(img_for_log.shape) == 3 and img_for_log.shape[0] in [1, 3, 4]:
            img_for_log = np.transpose(img_for_log, (1, 2, 0))
        
        # Handle data type - ensure it's in proper range for wandb
        if img_for_log.dtype != np.uint8:
            if img_for_log.max() <= 1.0:
                img_for_log = (img_for_log * 255).astype(np.uint8)
            else:
                img_for_log = img_for_log.astype(np.uint8)
        
        # Resize to consistent size (224x224) using PIL
        img_pil = Image.fromarray(img_for_log, mode='RGB')
        img_pil_resized = img_pil.resize((224, 224))
        img_for_log_resized = np.array(img_pil_resized)
        
        return img_for_log_resized

def log_simple_images_to_wandb(original_image, transformed_tensor, imgr_tensor, epoch, batch_idx, sample_idx, label, phase="train", log_frequency=50):
    """Simple logging of three images to wandb"""
    if batch_idx % log_frequency != 0:
        return
    
    # Only log first sample from each batch to avoid overwhelming wandb
    if sample_idx >= 1:
        return
    
    def tensor_to_numpy(tensor):
        """Convert tensor to numpy for wandb"""
        if tensor is None:
            return None
        
        img = tensor.clone().detach().cpu()
        
        # Denormalize if needed
        if img.min() < 0:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
        
        # Clamp and convert
        img = torch.clamp(img, 0, 1)
        img = (img * 255).byte().permute(1, 2, 0).numpy()
        return img
    
    # Log original image
    if original_image is not None:
        wandb.log({f"{phase}_original_e{epoch}_b{batch_idx}": wandb.Image(original_image, caption=f"Original - Label: {label}")})
    
    # Log transformed image
    transformed_np = tensor_to_numpy(transformed_tensor)
    if transformed_np is not None:
        wandb.log({f"{phase}_transformed_e{epoch}_b{batch_idx}": wandb.Image(transformed_np, caption=f"Transformed - Label: {label}")})
    
    # Log model output
    imgr_np = tensor_to_numpy(imgr_tensor)
    if imgr_np is not None:
        wandb.log({f"{phase}_imgr_e{epoch}_b{batch_idx}": wandb.Image(imgr_np, caption=f"Model Output - Label: {label}")})

def train_ucf_sports(train_loader, model, criterion, optimizer, epoch, use_cuda, scheduler):
    """Training function for UCF Sports Action dataset"""
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, batch_data in enumerate(train_loader):
        # Unpack the batch data - now includes original images as numpy arrays
        inputs, targets, original_numpy_images, original_wandb_images = batch_data
        
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs, imgr = model(inputs)
        targets = targets.long()        
        loss = criterion(outputs, targets)

        # Log three images simply
        for sample_idx in range(min(1, inputs.size(0))):  # Log first sample only
            log_simple_images_to_wandb(
                original_image=original_wandb_images[sample_idx],
                transformed_tensor=inputs[sample_idx],
                imgr_tensor=imgr[sample_idx] if imgr is not None else None,
                epoch=epoch,
                batch_idx=batch_idx,
                sample_idx=sample_idx,
                label=targets[sample_idx].item(),
                phase="train",
                log_frequency=50
            )

        #log to wandb
        if batch_idx == 200:
            print(f"Outputs: {outputs}")
            print(f"Targets: {targets}")
            print(f"Loss: {loss}\n")
                
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
        
        run.log({
            'epoch': epoch + 1,
            'train_loss': losses.avg,
            'train_acc': top1.avg,
            'train_top5': top5.avg,
        })
        
    bar.finish()
    return (losses.avg, top1.avg)

def test_ucf_sports(val_loader, model, criterion, epoch, use_cuda):
    """Testing function for UCF Sports Action dataset"""
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, batch_data in enumerate(val_loader):
        # Unpack the batch data - now includes original images as numpy arrays
        inputs, targets, original_numpy_images, original_wandb_images = batch_data
        
        # measure data loading time
        data_time.update(time.time() - end)

        # Convert targets to long immediately to fix uint32 issue
        targets = targets.long()

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        # Convert to Variables (deprecated but keeping for compatibility)
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs, imgr = model(inputs)
        
        # Log three images simply
        for sample_idx in range(min(1, inputs.size(0))):  # Log first sample only
            log_simple_images_to_wandb(
                original_image=original_wandb_images[sample_idx],
                transformed_tensor=inputs[sample_idx],
                imgr_tensor=imgr[sample_idx] if imgr is not None else None,
                epoch=epoch,
                batch_idx=batch_idx,
                sample_idx=sample_idx,
                label=targets[sample_idx].item(),
                phase="test",
                log_frequency=50
            )
        
        # Ensure targets are in valid range
        num_classes = outputs.shape[1]

        targets = torch.clamp(targets, 0, num_classes - 1)

        # Compute loss with error handling
        try:
            loss = criterion(outputs, targets)
        except RuntimeError as e:
            print(f"ERROR in loss computation at batch {batch_idx}: {e}")
            # Skip this batch
            continue

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        
        run.log({
            'epoch': epoch + 1,
            'test_loss': losses.avg,
            'test_acc': top1.avg,
            'test_top5': top5.avg,
        })
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))



if __name__ == '__main__':
    
    #get the arg of the currect dataset 
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
