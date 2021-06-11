from losses import *
from utils import *
from training_functions import *
from mc_dataset_infinite_patch3D import *
from convlstm3D import *
from stack_convlstm3D import *


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from glob import glob
from tqdm import tqdm
import os, sys, time, random, argparse
import SimpleITK as sitk
import datetime
from configparser import ConfigParser

# Parser parameters
args = parse_args()
# Get the cases paths
cases = list_cases(simu_path, exclude=[])
# Instantiate the selected model and indicate whether it's UNet based or not
model, unet = instantiate_model(args.model_name)
# Create train and validation dataloaders
train_loader, val_loader = create_dataloaders(args, cases, unet)
# Get the optimizer
optimizer = instantiate_optimizer(args, model)
# Get the learning rate scheduler
my_lr_scheduler = create_lr_scheduler(args, optimizer)
# Get the loss function
loss = create_loss(args)
# Write training configuration file
create_saving_framework(args)
# Instantitate tensorboard writer to write results for training monitoring
writer = SummaryWriter(save_path)


    
if args.mode == "infinite":
    print("Infinite training (mode: {})".format(args.mode))
    
    # Limited number of iterations
    iter_limit = int(6e5 / args.batch_size)


    count_no_improvement = 0
    best_val, best_train = np.inf, np.inf
    model.train()
    val_step = 10
    loss_train, ssim_train, mse_train, l1_train = 0, 0, 0, 0
    a = time.time()
    for iteration, data in enumerate(train_loader, 0):
        if iteration > iter_limit: 
            print("Stopped training at 1e5 iterations.")
            break

        sequence, target = data
        sequence = sequence.float().cuda()
        target = target.float().cuda()

        loss_, ssim_, mse_, l1_ = train(sequence, target, model, loss, optimizer, unet)

        loss_train += loss_ / val_step
        ssim_train += ssim_ / val_step
        mse_train  += mse_ / val_step
        l1_train   += l1_ / val_step


        # Validation step
        if iteration % val_step == 0:        
            loss_val, mse_val, ssim_val, l1_val, pred, gt = validate(model, loss, val_loader, n_val=n_val, unet=unet)

            # Decrease learning rate when needed
            if lr_scheduler == "plateau":
                my_lr_scheduler.step(loss_val)
            else:
                my_lr_scheduler.step()
            
            # Writing to tensorboard
            writer.add_scalars("Loss: {}".format(loss_name), {"train":loss_train, "validation":loss_val}, iteration)    
            writer.add_scalars("SSIM", {"train":ssim_train, "validation":ssim_val}, iteration)
            writer.add_scalars("MSE",  {"train":mse_train,  "validation":mse_val}, iteration)
            writer.add_scalars("L1",   {"train":l1_train,   "validation":l1_val}, iteration)
            writer.add_scalar("Learning rate", get_lr(optimizer), iteration)
            
            # Create figure of samples to visualize
            if iteration % 20 == 0:
                idx = int(target.shape[1] / 2)
                for k in range(len(pred)):
                    fig = plt.figure(figsize=(12, 6))
                    plt.subplot(121)
                    plt.title("Prediction")
                    plt.axis('off')
                    plt.imshow(pred[k, 0, idx], cmap="magma")
                    plt.subplot(122)
                    plt.title("Ground-truth")
                    plt.axis('off')
                    plt.imshow(gt[k, idx], cmap="magma")
                    writer.add_figure("Sample {}".format(k), fig, global_step=iteration, close=True)
            writer.flush()

            print("Iteration {} {:.2f} sec:\tLoss train:  {:.2e} \tLoss val:  {:.2e} \tL1 train: {:.2e} \tL1 val: {:.2e} \tSSIM train: {:.2e} \tSSIM val: {:.2e}".format(
                                                                                                                    iteration,
                                                                                                                    time.time() - a, 
                                                                                                                    loss_train, loss_val,
                                                                                                                    l1_train, l1_val,
                                                                                                                    ssim_train, ssim_val))
            # Save models when reaching new best on validation
            if loss_val < best_val: 
                count_no_improvement = 0
                best_val = loss_val
                torch.save({
                    'epoch': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, 
                    save_path + "/best_val_settings.pt")  
                torch.save(
                    model.state_dict(), 
                    save_path + "/best_val_model.pt")
            elif count_no_improvement > 5000:
                print("\nEarly stopping")
                break
            elif iteration > 500: 
                count_no_improvement += 1

            if loss_train < best_train:
                best_train = loss_train
                torch.save({
                    'epoch': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, 
                    save_path+ "/best_train_settings.pt")  
                torch.save(
                    model.state_dict(), 
                    save_path + "/best_train_model.pt")

            # Reset
            loss_train, ssim_train, mse_train, l1_train = 0, 0, 0, 0
            a = time.time()
        
        
else:
    print('Finite training (mode: {})'.format(args.mode))
    n_epochs = 100
    count_no_improvement = 0
    model.train()
    val_step = 10
    iter_limit = 1e5
    loss_train, ssim_train, mse_train, l1_train = 0, 0, 0, 0
    best_val, best_train = np.inf, np.inf
    a = time.time()
    
    for epoch in range(n_epochs):
        for iteration, data in enumerate(train_loader, 0):
            if iteration + len(train_loader) * epoch > iter_limit: break
            sequence, target = data
            sequence = sequence.float().cuda()
            target = target.float().cuda()

            loss_, ssim_, mse_, l1_ = train(sequence, target, model, loss, optimizer, unet)

            loss_train += loss_ / val_step
            ssim_train += ssim_ / val_step
            mse_train += mse_ / val_step
            l1_train += l1_ / val_step

            # Validation
            if iteration % val_step == 0:
                loss_val, mse_val, ssim_val, l1_val, pred, gt = validate(model, loss, val_loader, n_val=n_val, unet=unet)
                
                # Decrease learning rate when needed
                if lr_scheduler == "plateau":
                    my_lr_scheduler.step(loss_val)
                else:
                    my_lr_scheduler.step()

                writer.add_scalars("Loss: {}".format(loss_name), {"train":loss_train, "validation":loss_val}, iteration + len(train_loader) * epoch)    
                writer.add_scalars("SSIM", {"train":ssim_train, "validation":ssim_val}, iteration + len(train_loader) * epoch)
                writer.add_scalars("MSE", {"train":mse_train, "validation":mse_val}, iteration + len(train_loader) * epoch)
                writer.add_scalars("L1", {"train":l1_train, "validation":l1_val}, iteration + len(train_loader) * epoch)
                writer.add_scalar("Learning rate", get_lr(optimizer), iteration + len(train_loader) * epoch)

                if iteration % 20 == 0:
                    idx = int(args.patch_size / 2)
                    for k in range(len(pred)):
                        fig = plt.figure(figsize=(12, 6))
                        plt.subplot(121)
                        plt.title("Prediction")
                        plt.axis('off')
                        plt.imshow(pred[k, 0, idx], cmap="magma")
                        plt.subplot(122)
                        plt.title("Ground-truth")
                        plt.axis('off')
                        plt.imshow(gt[k, idx], cmap="magma")

                        writer.add_figure("Sample {}".format(k), fig, global_step=iteration + len(train_loader) * epoch, close=True)
                writer.flush()

                print("Iteration {} {:.2f} sec:\tLoss train:  {:.2e} \tLoss val:  {:.2e} \tL1 train: {:.2e} \tL1 val: {:.2e} \tSSIM train: {:.2e} \tSSIM val: {:.2e}".format(
                                                                                                                        iteration + len(train_loader) * epoch,
                                                                                                                        time.time() - a, 
                                                                                                                        loss_train, loss_val,
                                                                                                                        l1_train, l1_val,
                                                                                                                        ssim_train, ssim_val))
                # Save models
                if loss_val < best_val: 
                    count_no_improvement = 0
                    best_val = loss_val
                    torch.save({
                        'epoch': iteration + len(train_loader) * epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, 
                        save_path+ "/best_val_settings.pt")  
                    torch.save(
                        model.state_dict(), 
                        save_path + "/best_val_model.pt")
                elif count_no_improvement > 2000:
                    print("\nEarly stopping")
                    break
                else: 
                    count_no_improvement += 1

                if loss_train < best_train:
                    best_train = loss_train
                    torch.save({
                        'epoch': iteration + len(train_loader) * epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}, 
                        save_path+ "/best_train_settings.pt")  
                    torch.save(
                        model.state_dict(), 
                        save_path + "/best_train_model.pt")

                # Reset
                loss_train, ssim_train, mse_train, l1_train = 0, 0, 0, 0
                a = time.time()