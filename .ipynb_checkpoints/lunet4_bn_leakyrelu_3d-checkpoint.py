import torch
import torch.nn as nn
from convlstm3D import *
from copy import deepcopy

class DownBlock(nn.Module):    
    def __init__(self, in_channels, out_channels, to_bottleneck=False):
        super(DownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.to_bottleneck = to_bottleneck        
        self.conv1 = nn.Conv3d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm3d(self.out_channels)
        self.conv2 = nn.Conv3d(in_channels=self.out_channels,
                              out_channels=self.out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm3d(self.out_channels)
        self.maxpool = nn.MaxPool3d(kernel_size=2,
                                   stride=2)        
        self.clstm = ConvLSTM3DCell(input_dim=self.out_channels,
                                 hidden_dim=self.out_channels,
                                 kernel_size=(3, 3, 3),
                                 bias=True)
        
        
    def forward(self, input, cur_state):
        a1 = self.bn1(self.relu1(self.conv1(input)))
        a2 = self.bn2(self.relu2(self.conv2(a1)))
        h, c = self.clstm(a2, cur_state)
        if not self.to_bottleneck:
            going_down = self.maxpool(a2)
        else: going_down = a2
        # h will be concatenated with a skip connection
        return going_down, h, c

    
    
    
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__() 
        self.in_channels = in_channels
        self.out_channels = out_channels        
        self.conv1 = nn.Conv3d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm3d(self.out_channels)
        self.conv2 = nn.Conv3d(in_channels=self.out_channels,
                              out_channels=self.out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm3d(self.out_channels)
        
    def forward(self, input):
        a1 = self.bn1(self.relu1(self.conv1(input)))
        a2 = self.bn2(self.relu2(self.conv2(a1)))
        return a2
    
    
    
    
class UpBlock(nn.Module):    
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channels = in_channels        
        self.conv1 = nn.Conv3d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.bn1 = nn.BatchNorm3d(self.out_channels)
        self.deconv2 = nn.ConvTranspose3d(in_channels=self.out_channels,
                              out_channels=self.out_channels,
                              kernel_size=2,
                              stride=2,
                              padding=0) 
        self.bn2 = nn.BatchNorm3d(self.out_channels)
        
    def forward(self, input_sequence):
        a1 = self.bn1(self.conv1(input_sequence))
        a2 = self.bn2(self.deconv2(a1))
        return a2
    
    
    
def crop_weird_sizes(img):
    h, w = img.shape[-2], img.shape[-1]    
    if h % 2 != 0: img = img[..., :-1, :]
    if w % 2 != 0: img = img[..., :-1]
    return img



def adjust_crop(img_a, img_b):
    H, W = img_a.shape[-2], img_a.shape[-1]
    h_out, w_out = img_b.shape[-2], img_a.shape[-1]
        
    if h_out < H:
        diff = H - h_out
        img_a = img_a[..., :-diff, :]
    if w_out < W:
        diff = W - w_out
        img_a = img_a[..., :-diff]    
    
    try: assert img_b.shape[-2] == img_a.shape[-2]
    except: print("Dimension mismatch (height): input {} vs output {} vs output after crop {}".format(H, h_out, img_a.shape[-2]))
    try: assert img_b.shape[-1] == img_a.shape[-1]
    except: print("Dimension mismatch (width): input {} vs output {} vs output after crop {}".format(W, w_out, img_a.shape[-1]))
    return img_a



class LUNet4BNLeaky3D(nn.Module):
    def __init__(self, return_last=True):
        super(LUNet4BNLeaky3D, self).__init__()       
        self.model_name = "lunet4"
        self.return_last = return_last
        
        self.down1 = DownBlock(1, 64)
        self.down2 = DownBlock(64, 128) 
        self.down3 = DownBlock(128, 256) 
        self.down4 = DownBlock(256, 512, to_bottleneck=True)
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(512, 128)
        self.up3 = UpBlock(256, 1)
        
        self.Encoder = [self.down1, self.down2, self.down3, self.down4]
        self.Bottleneck = Bottleneck(512, 512)
        self.Decoder = [self.up1, self.up2, self.up3]
        
    def forward(self, input_sequence):
                
        # Initialize the lstm cells
        b, _, _, H, W, D = input_sequence.size()
        hidden_states = []
        for i, block in enumerate(self.Encoder):
            height, width, depth = H // (2**i), W // (2**i), D // (2**i)
#             height, width, depth = H, W // (2**i), D // (2**i)
            h_t, c_t = block.clstm.init_hidden(b, (height, width, depth))
            hidden_states += [(h_t, c_t)]
        
        # Forward 
        time_outputs = []
        seq_len = input_sequence.shape[1]
        for t in range(seq_len):
            skip_inputs = []
            frame = input_sequence[:, t, ...]
            
            # Forward through encoder
            for i, block in enumerate(self.Encoder):
                h_t, c_t = hidden_states[i]
                frame, h_t, c_t  = block(frame, [h_t, c_t])
                hidden_states[i] = (h_t, c_t)                   
                skip_inputs += [h_t]
                
            # We are at the bottleneck.
            bottleneck = self.Bottleneck(h_t)
            
            # Forward through decoder
            skip_inputs.reverse()
             
            for i, block in enumerate(self.Decoder):
                # Concat with skipconnections
                if i == 0:
                    decoded =  block(bottleneck)
                else:
                    skipped = skip_inputs[i]
                    concat = torch.cat([decoded, skipped], 1)
                    decoded = block(concat)
            
            if self.return_last and t == seq_len - 1: time_outputs = decoded
            elif not self.return_last: time_outputs += [decoded]
                 
        return time_outputs