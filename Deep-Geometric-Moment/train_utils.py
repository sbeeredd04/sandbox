import os
import shutil
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import wandb
from utils import AverageMeter, accuracy, Bar
import torchvision


def prepare_image_for_wandb(img_tensor):
    """
    Convert PyTorch tensor to format expected by wandb.Image
    Args:
        img_tensor: (C, H, W) tensor, typically in [0, 1] range
    Returns:
        numpy array in HWC format, uint8, range [0, 255]
    """
    img = img_tensor.detach().cpu()
    
    # Handle single channel by repeating to 3 channels
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    
    # Convert from CHW to HWC
    img = img.permute(1, 2, 0)
    
    # Normalize to [0, 1] if needed and convert to [0, 255] uint8
    if img.max() > 1.0:
        # Already in [0, 255] range
        img = img.clamp(0, 255).byte().numpy()
    else:
        # In [0, 1] range, scale to [0, 255]
        img = (img * 255).clamp(0, 255).byte().numpy()
    
    return img

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
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        
        # Handle video data: inputs shape is (batch_size, num_frames, C, H, W)
        # For testing, take the middle frame from each video sequence
        batch_size, num_frames, C, H, W = inputs.shape
        middle_frame = num_frames // 2
        inputs = inputs[:, middle_frame]  # Shape: (batch_size, C, H, W)
        
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

    # Create cumulative table for all predictions in this epoch
    columns = ["epoch", "batch_idx", "image_idx", "original_image", "reconstructed_image", "true_label", "predicted_label", "prediction_confidence"]
    train_predictions_table = wandb.Table(columns=columns)

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs, imgr = model(inputs)
        targets = targets.long()        
        loss = criterion(outputs, targets)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # Add predictions to cumulative table for each image in the batch
        for i in range(inputs.shape[0]):
            # Get predicted class and confidence
            predicted_class = torch.argmax(outputs[i]).item()
            prediction_confidence = torch.max(torch.softmax(outputs[i], dim=0)).item()
            
            # Convert tensors to wandb.Image format using our helper function
            original_img = wandb.Image(prepare_image_for_wandb(inputs[i]))
            reconstructed_img = wandb.Image(prepare_image_for_wandb(imgr[i]))
            
            # Add row to cumulative table
            train_predictions_table.add_data(
                epoch,
                batch_idx,
                i,
                original_img,
                reconstructed_img,
                targets[i].item(),
                predicted_class,
                f"{prediction_confidence:.3f}"
            )

        
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
        
        #log to wandb epoch metrics and cumulative predictions table
        wandb.log({
            'epoch': epoch,
            'batch_number': batch_idx,
            'train_loss': losses.avg,
            'train_top1': top1.avg,
            'train_top5': top5.avg,
        })

    wandb.log({
        'train_predictions': train_predictions_table
    })
    
    bar.finish()
    return (losses.avg, top1.avg)


def test_ucf_sports(val_loader, model, criterion, epoch, use_cuda):
    """Testing function for UCF Sports Action dataset"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # Create cumulative table for all predictions in this epoch
    columns = ["epoch", "batch_idx", "image_idx", "original_image", "reconstructed_image", "true_label", "predicted_label", "prediction_confidence"]
    test_predictions_table = wandb.Table(columns=columns)

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
        outputs, imgr = model(inputs)
        
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
        
        # Add predictions to cumulative table for each image in the batch
        for i in range(inputs.shape[0]):
            # Get predicted class and confidence
            predicted_class = torch.argmax(outputs[i]).item()
            prediction_confidence = torch.max(torch.softmax(outputs[i], dim=0)).item()
            
            # Convert tensors to wandb.Image format using our helper function
            original_img = wandb.Image(prepare_image_for_wandb(inputs[i]))
            reconstructed_img = wandb.Image(prepare_image_for_wandb(imgr[i]))
            
            # Add row to cumulative table
            test_predictions_table.add_data(
                epoch,
                batch_idx,
                i,
                original_img,
                reconstructed_img,
                targets[i].item(),
                predicted_class,
                f"{prediction_confidence:.3f}"
            )


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
        
        #log to wandb epoch metrics and cumulative predictions table
        wandb.log({
            'epoch': epoch,
            'batch_number': batch_idx,
            'test_loss': losses.avg,
            'test_top1': top1.avg,
            'test_top5': top5.avg,
        })
        
    wandb.log({
        'test_predictions': test_predictions_table
    })
        
    bar.finish()
    return (losses.avg, top1.avg)
