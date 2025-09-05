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

from model import ResNet18
from utils import Logger, mkdir_p, savefig
from train_utils import (
    WarmupCosineSchedule, save_checkpoint, train, test, 
    train_olympic_action, test_olympic_action, 
    train_ucf_sports, test_ucf_sports
)
from ucf_action_utils import UCFSportsDataset, get_ucf_sports_transforms

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

# Device options
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
        # Use custom dataset module
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
        
        # Test loader
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
        
        if args.dataset == 'olympic_action':
            train_loss, train_acc = train_olympic_action(train_loader, model, criterion, optimizer, epoch, use_cuda, scheduler)
            test_loss, test_acc = test_olympic_action(val_loader, model, criterion, epoch, use_cuda)
        elif args.dataset == 'ucf_sports':
            train_loss, train_acc = train_ucf_sports(train_loader, model, criterion, optimizer, epoch, use_cuda, scheduler)
            test_loss, test_acc = test_ucf_sports(val_loader, model, criterion, epoch, use_cuda)
        else:
            train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda, scheduler)
            test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)

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