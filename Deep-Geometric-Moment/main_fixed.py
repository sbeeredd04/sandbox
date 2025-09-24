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
# FIXED: Updated import to include CUDA-safe functions
from ucf_action_utils import UCFSportsDataset, get_ucf_sports_transforms, setup_multiprocessing_for_cuda, create_cuda_safe_dataloader
import deeplake
ds = deeplake.load('hub://activeloop/ucf-sports-action')

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
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--model-path', default='', type=str, metavar='PATH', help='path to model for inference (default: none)')
parser.add_argument('--image-path', default='', type=str, metavar='PATH', help='path to image for inference (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# FIXED: Setup multiprocessing for CUDA compatibility BEFORE any CUDA operations
print("Setting up multiprocessing for CUDA compatibility...")
setup_multiprocessing_for_cuda()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

if use_cuda:
    print(f"CUDA is available with {torch.cuda.device_count()} devices")
    print(f"Using GPU IDs: {args.gpu_id}")
else:
    print("CUDA is not available, using CPU")

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    # Handle inference mode
    if args.model_path and args.image_path:
        if os.path.isdir(args.image_path):
            print(f"Running batch inference on folder: {args.image_path}")
        else:
            print(f"Running inference on image: {args.image_path}")
        return

    # Data loading code
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
    elif args.dataset == 'imagenet':
        dataloader = datasets.ImageNet
        num_classes = 1000
    else:
        raise NotImplementedError("Dataset {} is not supported.".format(args.dataset))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_loader = None
    val_loader = None

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        # Data loading code for CIFAR
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        if args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
        else:
            train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)

        train_loader = data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    elif args.dataset == 'imagenet':
        # Data loading code for ImageNet
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        test_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

        train_loader = data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = data.DataLoader(test_dataset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    elif args.dataset == 'olympic_action':
        # Olympic Action dataset transforms
        from utils.olympic_action import OlympicActionDataset, get_olympic_action_transforms
        transform = get_olympic_action_transforms()
        
        # Create Olympic Action datasets
        trainset = OlympicActionDataset(ds, split='train', transform=transform)
        train_loader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        
        # Test loader
        testset = OlympicActionDataset(ds, split='test', transform=transform)
        val_loader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        
        # Get number of classes
        num_classes = trainset.get_num_classes()
        print(f"Olympic Action dataset loaded successfully with {num_classes} classes")
        
    elif args.dataset == 'ucf_sports':
        # UCF Sports Action dataset transforms
        transform = get_ucf_sports_transforms()
        
        # Create UCF Sports datasets with grouped classes
        trainset = UCFSportsDataset(ds, split='train', transform=transform, use_grouped_classes=True)
        # FIXED: Use CUDA-safe DataLoader creation
        train_loader = create_cuda_safe_dataloader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        
        # Test loader
        testset = UCFSportsDataset(ds, split='test', transform=transform, use_grouped_classes=True)
        # FIXED: Use CUDA-safe DataLoader creation
        val_loader = create_cuda_safe_dataloader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        
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
        model = DGMResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channels=3, input_height=224, input_width=224)
    elif args.dataset == 'olympic_action':
        # Olympic Action uses 224x224 images
        from model import DGMResNet, BasicBlock
        model = DGMResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, input_channels=3, input_height=224, input_width=224)
    else:
        # CIFAR and ImageNet models
        model = ResNet18(num_classes=num_classes)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    # Initialize wandb
    wandb.init(
        project="deep-geometric-moment",
        name=f"{args.dataset}_{args.epochs}epochs_{args.train_batch}batch",
        config=args
    )

    # Evaluate mode
    if args.evaluate:
        print('\nEvaluation mode')
        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))
        test_acc = test(val_loader, model, criterion, args)
        print('Test Acc: {:.3f}'.format(test_acc))
        return

    # Print model info
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # Create checkpoint directory
    mkdir_p(args.checkpoint)

    # Train and test
    for epoch in range(start_epoch, args.epochs):
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train for one epoch
        if args.dataset == 'olympic_action':
            train_acc = train_olympic_action(train_loader, model, criterion, optimizer, epoch, args)
        elif args.dataset == 'ucf_sports':
            train_acc = train_ucf_sports(train_loader, model, criterion, optimizer, epoch, args)
        else:
            train_acc = train(train_loader, model, criterion, optimizer, epoch, args)

        # Test on validation set
        if args.dataset == 'olympic_action':
            test_acc = test_olympic_action(val_loader, model, criterion, args)
        elif args.dataset == 'ucf_sports':
            test_acc = test_ucf_sports(val_loader, model, criterion, args)
        else:
            test_acc = test(val_loader, model, criterion, args)

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'learning_rate': current_lr
        })

        # Remember best acc and save checkpoint
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

        # Print statistics
        print('Epoch: [{0} | {1}] LR: {2:.6f}'.format(epoch + 1, args.epochs, current_lr))
        print('Train Acc: {:.3f} Test Acc: {:.3f}'.format(train_acc, test_acc))
        print('Best Test Acc: {:.3f}'.format(best_acc))
        print('-' * 80)

    print('Final best accuracy: {:.3f}'.format(best_acc))

if __name__ == '__main__':
    main()
