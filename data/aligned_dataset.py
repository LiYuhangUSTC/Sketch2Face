import os.path
import glob
from data.base_dataset import BaseDataset, get_params, get_transform, normalize, get_transform_sketch
from data.image_folder import make_dataset
from PIL import Image, ImageChops
import numpy as np
import torch

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (sketch)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A' )
        self.A_paths = sorted(make_dataset(self.dir_A))
        
        ### input B (photo)
        if self.opt.isTrain or self.opt.use_encoded_image:
            self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')  
            self.B_paths = sorted(make_dataset(self.dir_B))
            print('B_paths', len(self.B_paths))
            

        ### input C (deform sketch)
        self.dir_C = os.path.join(opt.dataroot, opt.phase + '_C')  
        self.C_list_paths = []
        for A_path in self.A_paths:
            basename = os.path.splitext(os.path.basename(A_path))[0]
            C_list_path = glob.glob(os.path.join(self.dir_C, '*', basename + '.png'))
            self.C_list_paths.append(C_list_path)

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):
        ### input A 
        A_path = self.A_paths[index]              
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        transform_A = get_transform_sketch(self.opt, params)
        A_tensor = transform_A(A.convert('RGB')) # A_tensor: [-1, 1]

        ### input B (real images)
        B_tensor = inst_tensor = feat_tensor = C_tensor = 0
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)
            
        ### input C 
        rand_index = np.random.randint(len(self.C_list_paths[index]))
        C_path = self.C_list_paths[index][rand_index] 
        C = Image.open(C_path).convert('RGB')
        transform_C = get_transform_sketch(self.opt, params)      
        C_tensor = transform_C(C)
 
        if self.opt.mix_sketch:
            rand_mix = np.random.randint(2)
        
        input_dict = {}
        input_dict['sketch'] =  C_tensor if self.opt.mix_sketch and rand_mix else A_tensor
        input_dict['photo'] =  B_tensor
        input_dict['sketch_deform'] = A_tensor if self.opt.mix_sketch and rand_mix else C_tensor 
        input_dict['path'] =  A_path

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset' 