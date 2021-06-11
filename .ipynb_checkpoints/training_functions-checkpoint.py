import torch
import pytorch_ssim


def validate(model, criterion, dataloader, n_val, unet=False):
    running_val_loss, running_mse_loss, running_ssim_loss, running_l1_loss = 0, 0, 0, 0    
    # Losses
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    ssim_loss = pytorch_ssim.SSIM()    
    # Validation
    count, count_batch =, 0 0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):

            sequence, target = data
            sequence = sequence.float().cuda()
            target = target.float().cuda()
            
            outputs = model(sequence)
            if unet: target = target[:, 0]
            else: target = target[:, 0, 0]
                
            # Training loss
            loss = criterion(outputs[:, 0], target)    
            
            # Evaluation losses
            mse_ = mse_loss(outputs[:, 0], target).item()
            ssim_ = ssim_loss(outputs[:, 0], target).item()
            l1 = l1_loss(outputs[:, 0], target).item()
            
            running_val_loss += loss.item()
            running_l1_loss += l1_
            running_mse_loss += mse_
            running_ssim_loss += ssim_          
         
            if count > n_val: break
            else: 
                count_batch += 1
                count += len(target)
               
    # Get the average loss per batch
    running_val_loss /= count_batch
    running_mse_loss /= count_batch
    running_ssim_loss /= count_batch
    running_l1_loss /= count_batch
    return running_val_loss, running_mse_loss, running_ssim_loss, running_l1_loss, outputs.detach().cpu().numpy()[:5], target.detach().cpu().numpy()[:5]

def train(sequence, target, model, loss, optimizer, unet=False):
    
    # Losses
    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_ssim.SSIM() 
    l1_loss = torch.nn.L1Loss()
    
    # Prediction
    outputs = model(sequence)
    
    if unet: target = target[:, 0]
    else: target = target[:, 0, 0]
    loss_value = loss(outputs[:, 0], target)

    # Backpropagation
    loss_value.backward()
    optimizer.step()
    optimizer.zero_grad() 

    # print statistics
    loss_ = loss_value.item()
    ssim_ = ssim_loss(outputs[:, 0], target).item()
    mse_ = mse_loss(outputs[:, 0], target).item()
    l1_ = l1_loss(outputs[:, 0], target).item()
    return loss_, ssim_, mse_, l1_