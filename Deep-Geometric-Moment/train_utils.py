import os
import shutil
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import wandb
from utils import AverageMeter, accuracy, Bar
import torchvision
import imageio
import cv2
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as F
from PIL import Image


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


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


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

        # compute output - model returns 5 values: cl, imgr1, imgr2, imgr3, imgr4
        outputs, imgr1, imgr2, imgr3, imgr4 = model(inputs)
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

        # compute output - model returns 5 values: cl, imgr1, imgr2, imgr3, imgr4
        outputs, imgr1, imgr2, imgr3, imgr4 = model(inputs)
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
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        # Handle video data: inputs shape is (batch_size, num_frames, C, H, W)
        # Take a random frame from each video sequence for training
        batch_size, num_frames, C, H, W = inputs.shape
        frame_indices = torch.randint(0, num_frames, (batch_size,))
        inputs = inputs[torch.arange(batch_size), frame_indices]  # Shape: (batch_size, C, H, W)
        
        # compute output - model returns 5 values: cl, imgr1, imgr2, imgr3, imgr4
        outputs, imgr1, imgr2, imgr3, imgr4 = model(inputs)
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
        if scheduler is not None:
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


def test_olympic_action(val_loader, model, criterion, epoch, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()
    
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        # Handle video data: inputs shape is (batch_size, num_frames, C, H, W)
        # For testing, take the middle frame from each video sequence
        batch_size, num_frames, C, H, W = inputs.shape
        middle_frame = num_frames // 2
        inputs = inputs[:, middle_frame]  # Shape: (batch_size, C, H, W)
        
        # compute output - model returns 5 values: cl, imgr1, imgr2, imgr3, imgr4
        outputs, imgr1, imgr2, imgr3, imgr4 = model(inputs)
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


def train_ucf_sports(train_loader, model, criterion, optimizer, epoch, use_cuda, scheduler, global_step=None):
    """Training function for UCF Sports Action dataset with preprocessed images"""
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # Use provided global_step or create a fallback
    if global_step is None:
        global_step = epoch * len(train_loader)

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        # Inputs already contain preprocessed images (OWL-ViT detection + SAM segmentation)
        # loaded from disk - no on-the-fly processing needed
        
        if batch_idx < 1:
            print(f"Preprocessed inputs shape: {inputs.shape}")
            # Get class name for first sample for debugging
            class_name = train_loader.dataset.get_class_name(targets[0].item())
            print(f"Class name: {class_name}")

        # compute output
        outputs, imgr1, imgr2, imgr3, imgr4 = model(inputs)
        targets = targets.long()        
        loss = criterion(outputs, targets)
            
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # Log images with step-based logging for first few batches
        if batch_idx < 2: 
            # Process each image in the batch
            for i in range(min(inputs.shape[0], 4)):  # Log max 4 images per batch
                current_step = global_step + batch_idx * inputs.shape[0] + i
                
                # Get predicted class and confidence
                predicted_class = torch.argmax(outputs[i]).item()
                prediction_confidence = torch.max(torch.softmax(outputs[i], dim=0)).item()
                true_class = targets[i].item()
                
                # Get class names
                true_class_name = train_loader.dataset.get_class_name(true_class)
                pred_class_name = train_loader.dataset.get_class_name(predicted_class)
                
                # Process images for logging
                # 1. Preprocessed input (denormalized)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = inputs[i].detach().cpu().numpy()
                inp = (inp * std[:, None, None]) + mean[:, None, None]
                inp = np.clip(inp, 0, 1)
                inp = np.transpose(inp, (1, 2, 0))  # C,H,W to H,W,C
                preprocessed_img = wandb.Image((inp * 255).astype(np.uint8), 
                                    caption=f"Preprocessed | True: {true_class_name} | Pred: {pred_class_name} | Conf: {prediction_confidence:.3f}")
                
                # 2-5. IMGR visualizations with viridis colormap
                img1 = imgr1[i, 0].detach().cpu().numpy()
                img_colored1 = plt.cm.viridis(img1)
                img_colored1 = (img_colored1[:, :, :3] * 255).astype(np.uint8)
                imgr_vis1 = wandb.Image(img_colored1, caption=f"IMGR1 | {true_class_name}")
                
                img2 = imgr2[i, 0].detach().cpu().numpy()
                img_colored2 = plt.cm.viridis(img2)
                img_colored2 = (img_colored2[:, :, :3] * 255).astype(np.uint8)
                imgr_vis2 = wandb.Image(img_colored2, caption=f"IMGR2 | {true_class_name}")
                
                img3 = imgr3[i, 0].detach().cpu().numpy()
                img_colored3 = plt.cm.viridis(img3)
                img_colored3 = (img_colored3[:, :, :3] * 255).astype(np.uint8)
                imgr_vis3 = wandb.Image(img_colored3, caption=f"IMGR3 | {true_class_name}")
                
                img4 = imgr4[i, 0].detach().cpu().numpy()
                img_colored4 = plt.cm.viridis(img4)
                img_colored4 = (img_colored4[:, :, :3] * 255).astype(np.uint8)
                imgr_vis4 = wandb.Image(img_colored4, caption=f"IMGR4 | {true_class_name}")
                
                # Log all images with step-based logging
                wandb.log({
                    "train_images": [preprocessed_img, imgr_vis1, imgr_vis2, imgr_vis3, imgr_vis4],
                    "train_true_class": true_class_name,
                    "train_predicted_class": pred_class_name,
                    "train_confidence": prediction_confidence,
                    "train_epoch": epoch,
                    "train_batch": batch_idx
                }, step=current_step)
        
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
        
        # Log epoch metrics 
        wandb.log({
            'epoch': epoch,
            'batch_number': batch_idx,
            'train_loss': losses.avg,
            'train_top1': top1.avg,
            'train_top5': top5.avg,
        })
    
    bar.finish()
    final_global_step = global_step + len(train_loader) * train_loader.batch_size
    return (losses.avg, top1.avg, final_global_step)


def test_ucf_sports(val_loader, model, criterion, epoch, use_cuda, global_step=None):
    """Testing function for UCF Sports Action dataset with preprocessed images"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # Use provided global_step or create a fallback
    if global_step is None:
        global_step = epoch * 10000  # Large offset to avoid conflicts

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        
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
        outputs, imgr1, imgr2, imgr3, imgr4 = model(inputs)
        
        # Ensure targets are in valid range
        num_classes = outputs.shape[1]
        targets = torch.clamp(targets, 0, num_classes - 1)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        
        # Log images with step-based logging for first few batches
        if batch_idx < 2: 
            # Process each image in the batch
            for i in range(min(inputs.shape[0], 4)):  # Log max 4 images per batch
                current_step = global_step + batch_idx * inputs.shape[0] + i
                
                # Get predicted class and confidence
                predicted_class = torch.argmax(outputs[i]).item()
                prediction_confidence = torch.max(torch.softmax(outputs[i], dim=0)).item()
                true_class = targets[i].item()
                
                # Get class names
                true_class_name = val_loader.dataset.get_class_name(true_class)
                pred_class_name = val_loader.dataset.get_class_name(predicted_class)
                
                # Process images for logging
                # 1. Preprocessed input (denormalized)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = inputs[i].detach().cpu().numpy()
                inp = (inp * std[:, None, None]) + mean[:, None, None]
                inp = np.clip(inp, 0, 1)
                inp = np.transpose(inp, (1, 2, 0))  # C,H,W to H,W,C
                preprocessed_img = wandb.Image((inp * 255).astype(np.uint8), 
                                    caption=f"Preprocessed | True: {true_class_name} | Pred: {pred_class_name} | Conf: {prediction_confidence:.3f}")
                
                # 2-5. IMGR visualizations with viridis colormap
                img1 = imgr1[i, 0].detach().cpu().numpy()
                img_colored1 = plt.cm.viridis(img1)
                img_colored1 = (img_colored1[:, :, :3] * 255).astype(np.uint8)
                imgr_vis1 = wandb.Image(img_colored1, caption=f"IMGR1 | {true_class_name}")
                
                img2 = imgr2[i, 0].detach().cpu().numpy()
                img_colored2 = plt.cm.viridis(img2)
                img_colored2 = (img_colored2[:, :, :3] * 255).astype(np.uint8)
                imgr_vis2 = wandb.Image(img_colored2, caption=f"IMGR2 | {true_class_name}")
                
                img3 = imgr3[i, 0].detach().cpu().numpy()
                img_colored3 = plt.cm.viridis(img3)
                img_colored3 = (img_colored3[:, :, :3] * 255).astype(np.uint8)
                imgr_vis3 = wandb.Image(img_colored3, caption=f"IMGR3 | {true_class_name}")
                
                img4 = imgr4[i, 0].detach().cpu().numpy()
                img_colored4 = plt.cm.viridis(img4)
                img_colored4 = (img_colored4[:, :, :3] * 255).astype(np.uint8)
                imgr_vis4 = wandb.Image(img_colored4, caption=f"IMGR4 | {true_class_name}")
                
                # Log all images with step-based logging
                wandb.log({
                    "test_images": [preprocessed_img, imgr_vis1, imgr_vis2, imgr_vis3, imgr_vis4],
                    "test_true_class": true_class_name,
                    "test_predicted_class": pred_class_name,
                    "test_confidence": prediction_confidence,
                    "test_epoch": epoch,
                    "test_batch": batch_idx
                }, step=current_step)


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
        
        # Log epoch metrics
        wandb.log({
            'epoch': epoch,
            'batch_number': batch_idx,
            'test_loss': losses.avg,
            'test_top1': top1.avg,
            'test_top5': top5.avg,
        })
        
    bar.finish()
    final_global_step = global_step + len(val_loader) * val_loader.batch_size
    return (losses.avg, top1.avg, final_global_step)
