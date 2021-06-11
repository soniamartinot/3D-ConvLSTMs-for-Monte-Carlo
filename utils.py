import os
import torch
from glob import glob
import argparse
from configparser import ConfigParser
import datetime
from mc_dataset_infinite_patch3D import *
from convlstm3D import *
from stack_convlstm3D import *
from losses import * 


def parse_args():
    parser = argparse.ArgumentParser(
        description='3D ConvLSTM training',
        add_help=True)   
    
    parser.add_argument("--n_samples", '-n',          default=0, type=float,  
                        help='Number of training samples')
    parser.add_argument("--patch_size", '-ps',        default=64, type=int, 
                        help='Size the height and width of a patch')
    parser.add_argument("--n_train", '-nt',           default=40, type=int,
                        help='Number of cases to use in training set')
    parser.add_argument("--model_name", '-m',         default='stack3D_deep', type=str, 
                        help='Name of the model')    
    parser.add_argument("--loss_name", '-l',          default='ssim-smoothl1', type=str, 
                        help='Name of the loss')
    parser.add_argument("--learning_rate", '-lr',     default=5e-5, type=float, 
                        help='Initial learning rate')
    parser.add_argument("--weight_decay",             default=1e-6, type=float, 
                        help='Initial weight decay')
    parser.add_argument("--optimizer_name",           default='adamw', type=str, 
                        help='Name of the optimizer')
    parser.add_argument("--save_path", '-save',       default='.', type=str, 
                        help="Path to save the model's weigths and results")
    parser.add_argument('--gpu_number', '-g',         default=0, type=int, 
                        help='GPU identifier (default 0)')
    parser.add_argument('--all_channels', '-ac',      default=False action='store_false', 
                        help='Whether to select patches from all views')
    parser.add_argument('--normalized_by_gt', '-ngt', default=True, action='store_true', 
                        help='Whether to normalize input data by ground-truth maximum')
    parser.add_argument("--standardize", '-st',       default=False, action='store_false', 
                        help='Whether to standardize input data')
    parser.add_argument("--uncertainty_thresh", '-u', default=0.2, type=float, 
                        help='Uncertainty threshold')
    parser.add_argument("--dose_thresh", '-dt',       default=0.2, type=float, 
                        help='Dose threshold')
    parser.add_argument("--n_frames", '-nf',          default=3, type=int, 
                        help='Number of frames in the input sequence')
    parser.add_argument("--batch_size", '-bs',        default=8, type=int, 
                        help='Batch size')
    parser.add_argument("--add_ct", '-ct',            default=False, action='store_false', 
                        help='Whether to add CT in input sequence')
    parser.add_argument("--ct_norm", '-nct',          default=True,  action='store_true', 
                        help='Whether to normalize the CT')
    parser.add_argument("--high_dose_only", '-hd',    default=False, action='store_false', 
                        help='Whether to train only on high dose regions')
    parser.add_argument("--p1", '-p1',                default=0.1, type=float, 
                        help="Probability below which you draw patches from low dose regions")
    parser.add_argument("--p2", '-p2',                default=0.6, type=float, 
                        help='Probability above which you draw patches from high dose regions')
    parser.add_argument("--single_frame", '-sf',      action='store_false', 
                        help='Whether you train on single frame instead of sequence')
    parser.add_argument("--mode", '-m',                 default='infinite', type=str, 
                        help="Whether to do finite or infinite training")
    parser.add_argument("--lr_scheduler", '-lrs',       default='plateau', type=str, 
                        help="Name of the learning rate scheduler to use")
    parser.add_argument("--depth", '-d',                default='64', type=int, 
                        help='Depth of an input patch')
    parser.add_argument("--raw", '-r',                  default=False, action='store_false', 
                        help='Whether to train on raw data with no preprocessing whatsoever')
    parser.add_argument("--n_layers", '-nl',            default=3, type=int,
                        help="Number of layers for BiONet3D")
    args = parser.parse_args()
    return args


def instantiate_model(args):

    if  args.model_name == "stack3D": 
        unet = False
        model = stack_model_3D()
    elif args.model_name == "stack3D_deep":             
        unet = False
        model = stack_model_3D_deep()
    elif args.model_name == "lunet4-bn-leaky3D":  
        model = LUNet4BNLeaky3D()
        unet = False
    elif args.model_name == "lunet4-bn-leaky3D_big":  
        model = LUNet4BNLeaky3D_big()
        unet = False  
    elif args.model_name == "bionet3d": 
        if args.single_frame and args.add_ct:       model = BiONet3D(input_channels=2, num_layers=args.num_layers)
        elif args.single_frame:                     model = BiONet3D(input_channels=1, num_layers=args.num_layers)
        elif not args.single_frame and args.add_ct: model = BiONet3D(input_channels=args.n_frames + 1, num_layers=args.num_layers)
        else:                                       model = BiONet3D(input_channels=args.n_frames, num_layers=args.num_layers)
        unet = True
    elif args.model_name == "unet3d":
        if args.single_frame and args.add_ct:       model = UNet3D(n_frames=2, num_layers=args.num_layers)
        elif args.single_frame:                     model = UNet3D(n_frames=1, num_layers=args.num_layers)
        elif not args.single_frame and args.add_ct: model = UNet3D(n_frames=args.n_frames + 1, num_layers=args.num_layers)
        else:                                       model = UNet3D(n_frames=args.n_frames, num_layers=args.num_layers)
        unet = True
    else: 
        print("Unrecognized model")
        break
    print("Model: ", args.model_name)
    return model, unet


def instantiate_optimizer(args, model):
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer_name == "adam":    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=wargs.eight_decay)
    elif args.optimizer_name == "adamw": optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    elif args.optimizer_name == "sgd":   optimizer = torch.optim.SGD(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else: print('Unrecognized optimizer', args.optimizer_name)
    return optimizer


def create_loss(args):
    if   args.loss_name == "mse":             loss = nn.MSELoss()
    elif args.loss_name == "l1":            loss = nn.L1Loss()
    elif args.loss_name == "l1smooth":      loss = nn.SmoothL1Loss()
    elif args.loss_name == "ssim":          loss = ssim_loss
    elif args.loss_name == "ssim-smoothl1": loss = ssim_smoothl1
    elif args.loss_name == "ssim-mse":      loss = ssim_mse
    print("Loss used: ", loss_name)
    return loss



def create_lr_scheduler(args, optimizer):
    # Learning rate decay
    if args.lr_scheduler == "plateau":
        decayRate = 0.8
        my_lr_scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                    mode='min', 
                                                                    factor=decayRate, 
                                                                    patience=15, 
                                                                    threshold=1e-2, 
                                                                    threshold_mode='rel', 
                                                                    cooldown=5, 
                                                                    min_lr=1e-7, 
                                                                    eps=1e-08, 
                                                                    verbose=True)
    elif args.lr_scheduler == "cosine":
    # Learning rate update: cosine anneeling
        my_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                     T_max=500, 
                                                                     eta_min=1e-8, 
                                                                     verbose=True)
    return my_lr_scheduler

def list_cases(path, exclude=[]):
    return [p for p in glob(path + "*") if len(os.path.basename(p)) == 4 and not os.path.basename(p) in exclude]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def create_saving_framework(args):
    # Don't forget to save all intermediate results and model
    now = str(datetime.now())
    training_name = "train_{}_{}".format(now.split(' ')[0], now.split(' ')[-1])
    os.system("mkdir {save_path}/{training_name}".format(save_path=args.save_path ,
                                                        training_name=training_name))
    save_path = args.save_path  + "/" + training_name
    print("Training:", training_name)
        
    parameters = {arg: getattr(args, arg) for arg in vars(args)}
    config_object = ConfigParser()
    config_object['Parameters for {}'.format(training_name)] = parameters

    # Write configuration file
    with open(save_path + '/config.ini', 'w') as conf:
        config_object.write(conf)
    
    
def create_dataloaders(args, cases, unet):
    n_train = args.n_train
    train = MC3DInfinitePatchDataset(cases[:n_train], 
                                     n_frames = args.n_frames, 
                                     patch_size = args.patch_size, 
                                     all_channels = args.all_channels, 
                                     normalized_by_gt = args.normalized_by_gt, 
                                     standardize = args.standardize,
                                     uncertainty_thresh = args.uncertainty_thresh, 
                                     dose_thresh = args.dose_thresh,
                                     unet = unet, 
                                     add_ct = args.add_ct, 
                                     ct_norm = args.ct_norm,
                                     high_dose_only = args.high_dose_only, 
                                     p1 = args.p1, 
                                     p2 = args.p2,
                                     single_frame = args.single_frame, 
                                     mode = args.mode, 
                                     n_samples = args.n_samples,
                                     depth = args.depth)

    val   = MC3DInfinitePatchDataset(cases[n_train:n_train+5], 
                                     n_frames = args.n_frames, 
                                     patch_size = args.patch_size, 
                                     all_channels = args.all_channels, 
                                     normalized_by_gt = args.normalized_by_gt, 
                                     standardize = args.standardize,
                                     uncertainty_thresh = args.uncertainty_thresh, 
                                     dose_thresh = args.dose_thresh,
                                     unet = unet, 
                                     add_ct = args.add_ct, 
                                     ct_norm = args.ct_norm,
                                     high_dose_only = args.high_dose_only, 
                                     p1 = args.p1, 
                                     p2 = args.p2,
                                     single_frame = args.single_frame, 
                                     mode = 'finite', 
                                     n_samples = args.n_samples,
                                     depth = args.depth)


    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val, batch_size=val_batch_size, shuffle=False, num_workers=6)
    return train_loader, val_loader