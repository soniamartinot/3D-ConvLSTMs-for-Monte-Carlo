import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import SimpleITK as sitk
import time, os, random
from tqdm import tqdm
from glob import glob
import multiprocessing
import matplotlib.pyplot as plt
from copy import copy



class MC3DInfinitePatchDataset(Dataset):
    def __init__(self, 
                 train_list, 
                 n_frames, 
                 ct_path,
                 patch_size=80, 
                 all_channels=False, 
                 normalized_by_gt=True, 
                 standardize=False, 
                 uncertainty_thresh=0.02, 
                 dose_thresh=0.8, 
                 seed=1, 
                 transform=True, 
                 unet=False, 
                 verbose=False, 
                 add_ct=False, 
                 ct_norm=False,
                 high_dose_only=False, 
                 p1=0.5, 
                 p2=0.2,
                 single_frame=False,
                 depth=8, 
                 mode="infinite",
                 n_samples=15000,
                 raw=False):
        
        # Set attributes
        self.train_list = train_list                   # list   - List comprising the paths to the cases used in the dataset
        self.n_frames = n_frames                       # int    - Number of noisy frames to input the model
        self.ct_path = ct_path                         # string - Path to CT images corresponding to cases in the train_list
        self.patch_size = patch_size                   # int    - Size of the patch
        self.all_channels = all_channels               # bool   - Whether to create patches with respect to each dimension
        self.normalized_by_gt = normalized_by_gt       # bool   - Whether to normalize the data beforehand
        self.standardize = standardize                 # bool   - Whether to standardize the data
        self.uncertainty_thresh = uncertainty_thresh   # float  - Set the uncertainty threshold below which we select the training samples
        self.dose_thresh = dose_thresh                 # float  - Set the dose threshold above which we select the training samples
        self.seed = seed                               # int    - Set the random seed
        self.transform = transform                     # bool   - Whether to add basic data augmentation
        self.unet = unet                               # bool   - Whether the model is unet like
        self.add_ct = add_ct                           # bool   - Whether to add the corresponding CT slice to the model's input
        self.ct_norm = ct_norm                         # bool   - Whether to normalize the CT volume
        self.high_dose_only = high_dose_only           # bool   - Whether to train on high dose regions only
        self.p1 = p1                                   # float  - Probability below which patches are drawn from low dose regions
        self.p2 = p2                                   # float  - Probability above which patches are drawn from high dose regions
        self.single_frame = single_frame               # bool   - Whether to train on a single frame instead of a sequence 
        self.depth = depth                             # int    - Depth of a patch
        self.mode = mode                               # bool   - Whether to train in "finite" (looping over the dataset) or "infinite" mode
        self.n_samples = n_samples                     # float  - Number of samples to train on 
        self.raw = raw                                 # bool   - Whether to train on raw data with no normalization whatsoever
        
        
        a = time.time()
        if verbose: print("\nLoading dataset...")        
        
        # If we want a variable size dataset
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
                
        # Particles to path dictionnary
        self.dict_particles = {case_path: self.get_particles_to_path(case_path) for case_path in tqdm(self.train_list)}
        self.dict_case_path = {os.path.basename(case_path): case_path for case_path in self.train_list}        
        self.dict_ct = {case_path: 
                        np.load(self.ct_path + "ct_{}.npy".format(os.path.basename(case_path)), allow_pickle=True) 
                        for case_path in tqdm(self.train_list)}
        
        
        
        if self.mode == "finite":
            self.path_idx_dict = {idx: random.choice(self.train_list) for idx in range(50)}
        
        if self.ct_norm:
            self.ct_max = 3071
            self.ct_min = -1000
            for case_path, ct in self.dict_ct.items():                
                self.dict_ct[case_path] = (ct - self.ct_min) / (self.ct_max - self.ct_min)
        
        if self.mode != "infinite":  
            print("Initialization of finite mode")
            # Here code hard mining when cases are too hard
            self.path_mapping = {idx: random.choice(self.train_list) for idx in range(self.n_samples)} 
            self.path_to_idx = {}
            for idx, case_path in self.path_mapping.items():
                self.path_to_idx[case_path] = self.path_to_idx.get(case_path, []) + [idx]  
                
            # Dictionnary mapping indexes to slice number
            if self.all_channels:    self.channel_mapping = {idx: np.random.randint(3) for idx in self.path_mapping}
            else:                    self.channel_mapping = {idx: 0 for idx in self.path_mapping}
            # init number of slices
            self.init_slice_numbers()  
        
        if verbose: print("Loading dataset with {} samples took:  {:.2f} minutes.\n".format(self.n_samples, (time.time() - a)/60))
            
            
    def __len__(self):
        if self.mode == 'infinite':  return int(1e6)    
        else: return self.n_samples
        
    def init_slice_numbers(self):    
        self.slice_mapping = {}
        for case_path, idx_list in tqdm(self.path_to_idx.items()):
            particles_to_path = self.dict_particles[case_path]
            particles = sorted(list(particles_to_path.keys()))  

            # Choose the slices where the uncertainty is the lowest
            case = os.path.basename(case_path)
            if os.path.isfile(case_path + "/{}_uncertainty_{}_0.npy".format(case, particles[-1])):
                relunc = np.load(case_path + "/{}_uncertainty_{}_0.npy".format(case, particles[-1]), allow_pickle=True)
            else:
                relunc = np.load(case_path + "/{}_uncertainty_{}.npy".format(case, particles[-1]), allow_pickle=True)

            # Choose where the dose is the highest
            dose = particles_to_path[particles[-1]][0] 

            # Probability
            p = np.random.rand() 
            if self.high_dose_only:
                if p > self.p1: thresh = 0.6
                elif self.p1 >= p > self.p2: thresh = 0.2
                else: thresh = 0
            else:       
                thresh = self.dose_thresh
            x_gt, y_gt, z_gt = np.where(dose > thresh * np.max(dose)) 
            x_unc, y_unc, z_unc = np.where(relunc < self.uncertainty_thresh)
            
            x_thresh = self.common_member(x_gt, x_unc)
            y_thresh = self.common_member(y_gt, y_unc)
            z_thresh = self.common_member(z_gt, z_unc)

            x_shape, y_shape, z_shape = dose.shape    
            half_patch_size = int(self.patch_size / 2)
            half_depth = int(self.depth / 2)
            for idx in idx_list:
                channel = self.channel_mapping[idx]
                if   channel == 0:
                    a = np.arange(half_depth, x_shape-half_depth)
                    b = np.arange(half_patch_size, y_shape-half_patch_size)
                    c = np.arange(half_patch_size, z_shape-half_patch_size)
                elif channel == 1:
                    a = np.arange(half_patch_size, x_shape-half_patch_size)
                    b = np.arange(half_depth, y_shape-half_depth)
                    c = np.arange(half_patch_size, z_shape-half_patch_size)
                elif channel == 2:
                    a = np.arange(half_patch_size, x_shape-half_patch_size)
                    b = np.arange(half_patch_size, y_shape-half_patch_size)
                    c = np.arange(half_depth, z_shape-half_depth)

                a = self.common_member(x_thresh, a)
                b = self.common_member(y_thresh, b) 
                c = self.common_member(z_thresh, c)
                self.slice_mapping[idx] = (np.random.randint(np.min(a), np.max(a)), 
                                           np.random.randint(np.min(b), np.max(b)),
                                           np.random.randint(np.min(c), np.max(c)))
                
                
    def get_particles_to_path(self, case_path):
        particles_to_path = {}
        for p in glob(case_path + "/*"):
            if not 'uncertainty' in p and not 'squared' in p:
                n = int(os.path.basename(p).split("/")[-1].split('_')[1].split('.')[0])
                particles_to_path[n] = particles_to_path.get(n, []) + [p]
        particles = sorted(list(particles_to_path))
        particles_to_path[particles[-1]] = [np.load(particles_to_path[particles[-1]][0], allow_pickle=True)]
        return particles_to_path   
    

    
    def create_pair(self, path, channel=0, idx=None, patch=True):
        particles_to_path = self.dict_particles[path]
        particles = sorted(list(particles_to_path))
        gt = particles_to_path[particles[-1]][0]
        
        # Get patch      
        half_patch_size = int(self.patch_size / 2)
        half_depth = int(self.depth / 2)
        
        if self.mode == "infinite":
            # Probability
            p = np.random.rand() 
            if self.high_dose_only:
                if p > self.p1: thresh = 0.6
                elif self.p1 >= p > self.p2: thresh = 0.2
                else: thresh = 0
            else:       
                if p > 0.5: thresh = 0.3
                else: thresh = 0.

            x_gt, y_gt, z_gt = np.where(gt >= np.max(gt) * thresh)

            x_shape, y_shape, z_shape = gt.shape
            if   channel == 0:
                a = np.arange(half_depth, x_shape-half_depth)
                b = np.arange(half_patch_size, y_shape-half_patch_size)
                c = np.arange(half_patch_size, z_shape-half_patch_size)
            elif channel == 1:
                a = np.arange(half_patch_size, x_shape-half_patch_size)
                b = np.arange(half_depth, y_shape-half_depth)
                c = np.arange(half_patch_size, z_shape-half_patch_size)
            elif channel == 2:
                a = np.arange(half_patch_size, x_shape-half_patch_size)
                b = np.arange(half_patch_size, y_shape-half_patch_size)
                c = np.arange(half_depth, z_shape-half_depth)


            a = self.common_member(x_gt, a)
            b = self.common_member(y_gt, b) 
            c = self.common_member(z_gt, c)


            # Get slice numbers        
            x = random.randint(np.min(a), np.max(a))
            y = random.randint(np.min(b), np.max(b))
            z = random.randint(np.min(c), np.max(c)) 
        
        elif idx is not None:
            # Get slice_number
            x, y, z = self.slice_mapping[idx]
        
        if patch:
            # Get ground-truth   
            if   channel == 0: ground_truth = copy(gt[x-half_depth:x+half_depth, y-half_patch_size:y+half_patch_size, z-half_patch_size:z+half_patch_size])
            elif channel == 1: ground_truth = copy(gt[x-half_patch_size:x+half_patch_size, y-half_depth:y+half_depth, z-half_patch_size:z+half_patch_size])
            elif channel == 2: ground_truth = copy(gt[x-half_patch_size:x+half_patch_size, y-half_patch_size:y+half_patch_size, z-half_depth:z+half_depth])     
            h, w, d = ground_truth.shape
            
            # If only a single input frame for example in the case of UNet
            if self.single_frame:
                # Create sequence with added CT in first place
                if self.add_ct:
                    sequence = np.empty((2, h, w, d))
                    n_particles = particles[int(self.n_frames -1)]
                    ind = np.random.randint(len(particles_to_path[n_particles]))
                    path = particles_to_path[n_particles][ind]
                    if   channel == 0: 
                        sequence[1] = np.load(path, allow_pickle=True)[x-half_depth:x+half_depth, y-half_patch_size:y+half_patch_size, z-half_patch_size:z+half_patch_size]
                    elif channel == 1: 
                        sequence[1] = np.load(path, allow_pickle=True)[x-half_patch_size:x+half_patch_size, y-half_depth:y+half_depth, z-half_patch_size:z+half_patch_size]
                    elif channel == 2: 
                        sequence[1] = np.load(path, allow_pickle=True)[x-half_patch_size:x+half_patch_size, y-half_patch_size:y+half_patch_size, z-half_depth:z+half_depth]
                    case = os.path.basename(path).split("_")[0]
                    sequence[0] = self.dict_ct[self.dict_case_path[case]][x-half_depth:x+half_depth, y-half_patch_size:y+half_patch_size, z-half_patch_size:z+half_patch_size]
                # Create sequence without CT
                else:
                    sequence = np.empty((1, h, w, d))
                    n_particles = particles[int(self.n_frames -1)]
                    ind = np.random.randint(len(particles_to_path[n_particles]))
                    path = particles_to_path[n_particles][ind]
                    if   channel == 0: 
                        sequence[0] = np.load(path, allow_pickle=True)[x-half_depth:x+half_depth, y-half_patch_size:y+half_patch_size, z-half_patch_size:z+half_patch_size]
                    elif channel == 1: 
                        sequence[0] = np.load(path, allow_pickle=True)[x-half_patch_size:x+half_patch_size, y-half_depth:y+half_depth, z-half_patch_size:z+half_patch_size]
                    elif channel == 2: 
                        sequence[0] = np.load(path, allow_pickle=True)[x-half_patch_size:x+half_patch_size, y-half_patch_size:y+half_patch_size, z-half_depth:z+half_depth]
                    
            # If several input frames
            else:               
                # Create sequence with added CT in first place
                if self.add_ct:
                    sequence = np.empty((self.n_frames+1, h, w, d))
                    for i, n in enumerate(particles[:self.n_frames]):
                        ind = np.random.randint(len(particles_to_path[n]))
                        path = particles_to_path[n][ind]
                        if   channel == 0: 
                            frame = np.load(path, allow_pickle=True)[x-half_depth:x+half_depth, y-half_patch_size:y+half_patch_size, z-half_patch_size:z+half_patch_size]
                        elif channel == 1: 
                            frame = np.load(path, allow_pickle=True)[x-half_patch_size:x+half_patch_size, y-half_depth:y+half_depth, z-half_patch_size:z+half_patch_size]
                        elif channel == 2: 
                            frame = np.load(path, allow_pickle=True)[x-half_patch_size:x+half_patch_size, y-half_patch_size:y+half_patch_size, z-half_depth:z+half_depth]
                        sequence[i+1] = frame
                    case = os.path.basename(path).split("_")[0]
                    sequence[0] = self.dict_ct[self.dict_case_path[case]][x-half_depth:x+half_depth, y-half_patch_size:y+half_patch_size, z-half_patch_size:z+half_patch_size]
                # Create sequence without CT
                else:
                    sequence = np.empty((self.n_frames, h, w, d))
                    for i, n in enumerate(particles[:self.n_frames]):
                        ind = np.random.randint(len(particles_to_path[n]))
                        path = particles_to_path[n][ind]
                        if   channel == 0: 
                            frame = np.load(path, allow_pickle=True)[x-half_depth:x+half_depth, y-half_patch_size:y+half_patch_size, z-half_patch_size:z+half_patch_size]
                        elif channel == 1: 
                            frame = np.load(path, allow_pickle=True)[x-half_patch_size:x+half_patch_size, y-half_depth:y+half_depth, z-half_patch_size:z+half_patch_size]
                        elif channel == 2: 
                            frame = np.load(path, allow_pickle=True)[x-half_patch_size:x+half_patch_size, y-half_patch_size:y+half_patch_size, z-half_depth:z+half_depth]
                        sequence[i] = frame
        else:
            ground_truth = gt
            h, w, d = ground_truth.shape
            sequence = np.empty((self.n_frames, h, w, d))
            for i, n in enumerate(particles[:self.n_frames]):
                ind = np.random.randint(len(particles_to_path[n]))
                path = particles_to_path[n][ind]
                sequence[i] = np.load(path, allow_pickle=True)            

        # Reshape
        a, b, c, d = sequence.shape
        sequence = sequence.reshape((a, 1, b, c, d)) 
        ground_truth = ground_truth.reshape((1, 1, b, c, d))     
        # Normalize by the max dose of the complete sequence (including ground truth)
        m = np.max(ground_truth)
        if self.normalized_by_gt: 
            print("Normalized by gt")
            sequence /= m
            ground_truth /= m
        # Else, scale between -1 and 1
        elif self.standardize: 
            sequence = (sequence - np.mean(sequence)) / np.std(sequence)
            ground_truth = (ground_truth - np.mean(ground_truth)) / np.std(ground_truth)
        # Raw data
        elif self.raw:
            sequence = sequence
            ground_truth = ground_truth
        # Else put every frame between 0 and 1
        else:   
            sequence /= np.ndarray.max(sequence, axis=(2, 3, 4))[:, np.newaxis, np.newaxis, np.newaxis]
            ground_truth /= m
        return sequence, ground_truth
    
    
    
    def common_member(self, a, b): 
        a_set = set(a) 
        b_set = set(b) 

        if (a_set & b_set): 
            return list(a_set & b_set) 
        else: 
            print("No common elements")  
            return b


    
    def crop_and_adapt(self, img):
        r_h, r_w, r_d = None, None, None
        H, W, D = img.shape[-3], img.shape[-2], img.shape[-1]
        if H % 2**3 != 0: r_h = - (H % 2**3)
        if W % 2**3 != 0: r_w = - (W % 2**3)
        if D % 2**3 != 0: r_d = - (D % 2**3)
        return img[..., :r_h, :r_w, :r_d]
        
        
    def __getitem__(self, idx):
        
        if self.mode == "infinite":
            # Get path to random case
            path = random.choice(self.train_list) 
        else:
            # Get path of precise case
#             path = self.path_idx_dict[idx]
#             path = random.choice(self.train_list) 
            path = self.path_mapping[idx]
            
        # Get sequence et the frame to predict        
        sequence, next_frame = self.create_pair(path, patch=True, idx=idx,
                                                channel=0)   
        
        # Turn into tensors
        sequence = torch.from_numpy(sequence)
        next_frame = torch.from_numpy(next_frame)
        
        # Apply transformations
        p = np.random.rand()
        if self.transform and p > 0.5:
            torch.manual_seed(idx)
            composed = transforms.Compose([transforms.RandomHorizontalFlip(p=1),
                                           transforms.RandomVerticalFlip(p=1)])
            
            # Concat to transform
            all_seq = torch.cat([sequence, next_frame], axis=0)
            all_seq = composed(all_seq)
            sequence = all_seq[:-1]
            next_frame = all_seq[-1:]
        
        if self.unet:
            # Crop to be processed by UNet
            sequence = self.crop_and_adapt(sequence)
            next_frame = self.crop_and_adapt(next_frame)
            t, c, h, w, d = sequence.shape
            sequence = torch.reshape(sequence, (t, h, w, d))
            next_frame = torch.reshape(next_frame, (1, h, w, d))
        return sequence.float(), next_frame.float()
    
    
    def get_volumes(self, case_path):
        particles_to_path = self.dict_particles[case_path]
        particles = sorted(list(particles_to_path.keys()))
        sequence, gt = self.create_pair(case_path, 
                                        channel=0, 
                                        patch=False)
        sequence = torch.from_numpy(sequence)
        sequence = self.crop_and_adapt(sequence)
        gt = self.crop_and_adapt(gt)
        H, W, D = sequence.shape[-3], sequence.shape[-2], sequence.shape[-1]
        
        if self.unet and not self.single_frame:
            sequence = self.crop_and_adapt(sequence)
            gt = self.crop_and_adapt(gt)
            t, _, h, w, d = sequence.shape
            sequence = torch.reshape(sequence, (t, h, w, d))
        elif self.unet and self.single_frame:
#             sequence = self.crop_and_adapt(sequence)[-1]
#             gt = self.crop_and_adapt(gt)
            sequence = sequence[-1]
            _, h, w, d = sequence.shape
            sequence = torch.reshape(sequence, (1, h, w, d))
        
        gt = np.reshape(gt, (H, W, D))
        return sequence, gt