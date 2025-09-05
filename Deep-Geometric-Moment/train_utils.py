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
import imageio
import cv2
import matplotlib.pyplot as plt

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
    columns = ["epoch", "batch_idx", "image_idx", "original_image", "transformed_image", "imgr_visualization", "true_label", "predicted_label", "prediction_confidence"]
    train_predictions_table = wandb.Table(columns=columns)

    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets, image) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs, imgr = model(inputs)
        targets = targets.long()        
        loss = criterion(outputs, targets)
        
                
        if batch_idx < 1: 
            print(f"="*100)
            print(f"Shape of inputs: {inputs.shape}")
            print(f"Type: {inputs.type()}")
            print(f"Min of inputs: {inputs.min()}")
            print(f"Max of inputs: {inputs.max()}")
            print(f"="*10)
            print(f"Shape of imgr: {imgr.shape}")
            print(f"Type: {imgr.type()}")
            print(f"Min of imgr: {imgr.min()}")
            print(f"Max of imgr: {imgr.max()}")
            print(f"="*10)
            print(f"Shape of copy_image: {image.shape}")
            print(f"Type: {image.type()}")
            print(f"Min of copy_image: {image.min()}")
            print(f"Max of copy_image: {image.max()}")
            print(f"="*100)
            
            #visualize the copy image 256 256 3
            for i in range(image.shape[0]):
                img_np = image[i].cpu().numpy()
                imageio.imwrite(f'./data/debug/original_image_{i}.png', img_np)
                        
            # Save imgr using matplotlib for better grayscale visualization
            for i in range(imgr.shape[0]):
                img = imgr[i, 0].detach().cpu().numpy()
                # Apply viridis colormap and rescale to uint8
                img_colored = plt.cm.viridis(img)
                img_colored = (img_colored[:, :, :3] * 255).astype(np.uint8)
                imageio.imwrite(f'./data/debug/imgr_viridis_{i}.png', img_colored)
                
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            for i in range(inputs.shape[0]):
                inp = inputs[i].detach().cpu().numpy()
                # de-normalize
                inp = (inp * std[:, None, None]) + mean[:, None, None]
                inp = np.clip(inp, 0, 1)
                inp = np.transpose(inp, (1, 2, 0))  # C,H,W to H,W,C
                inp = (inp * 255).astype(np.uint8)
                imageio.imwrite(f'./data/debug/transformed_image_{i}.png', inp)
            
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if batch_idx < 4: 
            # Add predictions to cumulative table for each image in the batch
            for i in range(inputs.shape[0]):
                # Get predicted class and confidence
                predicted_class = torch.argmax(outputs[i]).item()
                prediction_confidence = torch.max(torch.softmax(outputs[i], dim=0)).item()
                
                # Process images same as debug visualization
                # 1. Original image
                original_img = wandb.Image(image[i].cpu().numpy())
                
                # 2. Transformed (denormalized) input image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = inputs[i].detach().cpu().numpy()
                inp = (inp * std[:, None, None]) + mean[:, None, None]
                inp = np.clip(inp, 0, 1)
                inp = np.transpose(inp, (1, 2, 0))  # C,H,W to H,W,C
                transformed_img = wandb.Image((inp * 255).astype(np.uint8))
                
                # 3. imgr visualization with viridis colormap
                img = imgr[i, 0].detach().cpu().numpy()
                img_colored = plt.cm.viridis(img)
                img_colored = (img_colored[:, :, :3] * 255).astype(np.uint8)
                imgr_visualization = wandb.Image(img_colored)
                
                # Add row to cumulative table
                train_predictions_table.add_data(
                    epoch,
                    batch_idx,
                    i,
                    original_img,
                    transformed_img,
                    imgr_visualization,
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
    columns = ["epoch", "batch_idx", "image_idx", "original_image", "transformed_image", "imgr_visualization", "true_label", "predicted_label", "prediction_confidence"]
    test_predictions_table = wandb.Table(columns=columns)

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets, image) in enumerate(val_loader):
        
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
        loss = criterion(outputs, targets)
        
        if batch_idx < 1: 
            print(f"="*100)
            print(f"Shape of inputs: {inputs.shape}")
            print(f"Type: {inputs.type()}")
            print(f"Min of inputs: {inputs.min()}")
            print(f"Max of inputs: {inputs.max()}")
            print(f"="*10)
            print(f"Shape of imgr: {imgr.shape}")
            print(f"Type: {imgr.type()}")
            print(f"Min of imgr: {imgr.min()}")
            print(f"Max of imgr: {imgr.max()}")
            print(f"="*10)
            print(f"Shape of copy_image: {image.shape}")
            print(f"Type: {image.type()}")
            print(f"Min of copy_image: {image.min()}")
            print(f"Max of copy_image: {image.max()}")
            print(f"="*100)
            
            #visualize the copy image 256 256 3
            for i in range(image.shape[0]):
                img_np = image[i].cpu().numpy()
                imageio.imwrite(f'./data/debug/original_image_{i}.png', img_np)
                        
            # Save imgr using matplotlib for better grayscale visualization
            for i in range(imgr.shape[0]):
                img = imgr[i, 0].detach().cpu().numpy()
                # Apply viridis colormap and rescale to uint8
                img_colored = plt.cm.viridis(img)
                img_colored = (img_colored[:, :, :3] * 255).astype(np.uint8)
                imageio.imwrite(f'./data/debug/imgr_viridis_{i}.png', img_colored)
                
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            for i in range(inputs.shape[0]):
                inp = inputs[i].detach().cpu().numpy()
                # de-normalize
                inp = (inp * std[:, None, None]) + mean[:, None, None]
                inp = np.clip(inp, 0, 1)
                inp = np.transpose(inp, (1, 2, 0))  # C,H,W to H,W,C
                inp = (inp * 255).astype(np.uint8)
                imageio.imwrite(f'./data/debug/transformed_image_{i}.png', inp)
            

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        
        if batch_idx < 4: 
            # Add predictions to cumulative table for each image in the batch
            for i in range(inputs.shape[0]):
                # Get predicted class and confidence
                predicted_class = torch.argmax(outputs[i]).item()
                prediction_confidence = torch.max(torch.softmax(outputs[i], dim=0)).item()
                
                # Process images same as debug visualization
                # 1. Original image
                original_img = wandb.Image(image[i].cpu().numpy())
                
                # 2. Transformed (denormalized) input image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = inputs[i].detach().cpu().numpy()
                inp = (inp * std[:, None, None]) + mean[:, None, None]
                inp = np.clip(inp, 0, 1)
                inp = np.transpose(inp, (1, 2, 0))  # C,H,W to H,W,C
                transformed_img = wandb.Image((inp * 255).astype(np.uint8))
                
                # 3. imgr visualization with viridis colormap
                img = imgr[i, 0].detach().cpu().numpy()
                img_colored = plt.cm.viridis(img)
                img_colored = (img_colored[:, :, :3] * 255).astype(np.uint8)
                imgr_visualization = wandb.Image(img_colored)
                
                # Add row to cumulative table
                test_predictions_table.add_data(
                    epoch,
                    batch_idx,
                    i,
                    original_img,
                    transformed_img,
                    imgr_visualization,
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
