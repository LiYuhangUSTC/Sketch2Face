import os
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

#### for computing complexity
#from thop import profile
#from thop import clever_format

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_gfm, use_edge_penalty):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, use_gfm, use_edge_penalty)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, g_gfm, edge_penalty):
            return [l for (l,f) in zip((g_gan, g_gan_feat, g_vgg,d_real, d_fake, g_gfm, edge_penalty),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.input_nc

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc 

        print(opt.netG)
        if 'dual' in opt.netG:
            self.netG, self.netG_sap = networks.define_G(netG_input_nc, opt, gpu_ids=self.gpu_ids)        
        else:
            self.netG = networks.define_G(netG_input_nc, opt, gpu_ids=self.gpu_ids)
            
        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc 
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        # load networks
        if self.isTrain and opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            if 'dual' in opt.netG:
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
                self.load_network(self.netG_sap, 'G', opt.which_epoch, pretrained_path)
            else:
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
                if self.isTrain:
                    self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  


        if not self.isTrain or opt.continue_train:           
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            if 'dual' in opt.netG:
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
                self.load_network(self.netG_sap, 'G_sap', opt.which_epoch, pretrained_path)
            else:
                self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            
            if opt.continue_train:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  


        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and torch.cuda.device_count() > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, opt.gfm, opt.use_edge_penalty)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake', 'G_GFM', 'edge_penalty')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0 and 'dual'in opt.netG:                
                finetune_list = set()
                params_dict = dict(self.netG_sap.named_parameters())
                params_sap = []
                for key, value in params_dict.items():       
                    if key.startswith('max_pool') or key.startswith('model_'):                    
                        params_sap += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the SAP (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))     
                self.optimizer_G_sap = torch.optim.Adam(params_sap, lr=opt.lr, betas=(opt.beta1, 0.999))                            
            else:
                if 'dual' in opt.netG:
                    params = list(self.netG_sap.parameters()) + list(self.netG.parameters())
                    self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
                else:
                    params = list(self.netG.parameters())
                    self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, sketch, photo=None, deform_sketch=None):             
        # get edges from instance map
        sketch = Variable(sketch.data.cuda())

        # real images for training
        if photo is not None:
            photo = Variable(photo.cuda())
        # deformed sketch
        if deform_sketch is not None:
            deform_sketch = Variable(deform_sketch.cuda())

        return sketch, photo, deform_sketch

    def discriminate(self, sketch, photo, use_pool=False):
        '''
        * setup detach (detach tensor from the graph)
        * setup fake image pool for fake image
        '''
        input_concat = torch.cat((sketch, photo.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, sketch, photo, sketch_deform=None, infer=False):
        # Encode Inputs
        sketch, real_image, sketch_deform = self.encode_input(sketch, photo, sketch_deform)  

        return_dict = {}

        if self.opt.deform:
            fake_image_deform, gen_features_deform = self.netG_sap.forward(sketch_deform)      
            return_dict['fake_image_deform'] = fake_image_deform
        
        fake_image, gen_features = self.netG.forward(sketch)
        return_dict['fake_image'] = fake_image
        
        ########### compute loss for sketch ###################
        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(sketch, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)   
        
        # Real Detection and Loss        
        pred_real = self.discriminate(sketch, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        # Do not detach, so that the gradients pass through the generator
        pred_fake = self.netD.forward(torch.cat((sketch, fake_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # Discriminator feature matching loss
        loss_D_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_D_GAN_Feat += D_weights * feat_weights \
                                        * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) \
                                        * self.opt.lambda_feat
        
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat       
            

        ########### compute loss for sketch_deform ###################
        if self.opt.deform:
            ''' Same to sketch, but use sketch_deform'''
            pred_fake_pool_deform = self.discriminate(sketch_deform, fake_image_deform, use_pool=True)
            loss_D_fake_deform = self.criterionGAN(pred_fake_pool_deform, False)        
            pred_real_deform = self.discriminate(sketch_deform, real_image)
            loss_D_real_deform = self.criterionGAN(pred_real_deform, True)
            pred_fake_deform = self.netD.forward(torch.cat((sketch_deform, fake_image), dim=1))        
            loss_G_GAN_deform = self.criterionGAN(pred_fake_deform, True)               
            loss_D_GAN_Feat_deform = 0
            if not self.opt.no_ganFeat_loss:
                feat_weights = 4.0 / (self.opt.n_layers_D + 1)
                D_weights = 1.0 / self.opt.num_D
                for i in range(self.opt.num_D):
                    for j in range(len(pred_fake[i])-1):
                        loss_D_GAN_Feat_deform += D_weights * feat_weights \
                                            * self.criterionFeat(pred_fake_deform[i][j], pred_real_deform[i][j].detach()) \
                                            * self.opt.lambda_feat
            # Generator feature matching loss
            loss_G_GAN_Feat = 0
            loss_G_GAN_Feat_deform = 0
            if not self.opt.no_gen_ganFeat_loss:
                ###### asign Q for generator feature matching loss ########
                gen_feat_layer = self.opt.gfm_layers
                feat_weights = 1 / len(gen_feat_layer)
                for layer_index in gen_feat_layer:
                    loss_G_GAN_Feat +=  feat_weights \
                                        * self.criterionFeat(gen_features_deform[layer_index+2], gen_features[layer_index]) \
                                        * self.opt.lambda_feat

        
            if not self.opt.no_vgg_loss:
                loss_G_VGG_deform = self.criterionVGG(fake_image_deform, real_image) * self.opt.lambda_feat
                
            loss_edge_penalty = 0
            if self.opt.use_edge_penalty:
                pool_output_ = (gen_features[0] + 1.) / 2.  # set edge to one, background to zero
                pool_output_deform_ = (gen_features_deform[0] + 1.) / 2.  # set edge to one, background to zero
                sketch_ = (sketch + 1.) / 2.                # set edge to one, background to zero
                torchvision.utils.save_image(torchvision.utils.make_grid(pool_output_), 'pool_output_.png')
                torchvision.utils.save_image(torchvision.utils.make_grid(sketch_), 'sketch_.png')

                ep1 = pool_output_deform_ * (1. - sketch_)
                ep2 = sketch_ * (1. - pool_output_deform_)
                ep3 = sketch_ * (1. - pool_output_)
                torchvision.utils.save_image(torchvision.utils.make_grid(ep1), 'ep1.png')
                torchvision.utils.save_image(torchvision.utils.make_grid(ep2), 'ep2.png')
                torchvision.utils.save_image(torchvision.utils.make_grid(ep3), 'ep3.png')
                print(torch.mean(ep1) * self.opt.lambda_ep1, torch.mean(ep2) * self.opt.lambda_ep2, torch.mean(ep3) * self.opt.lambda_ep3, 'xxxxxxxxxxx ep')
                
                
                loss_edge_penalty += torch.mean(ep1) * self.opt.lambda_ep1
                loss_edge_penalty += torch.mean(ep2) * self.opt.lambda_ep2
                loss_edge_penalty += torch.mean(ep3) * self.opt.lambda_ep3
                
                

        _loss_G_GAN = 0.5 * (loss_G_GAN + loss_G_GAN_deform) if self.opt.deform else loss_G_GAN
        _loss_GAN_Feat = 0.5 * (loss_D_GAN_Feat + loss_D_GAN_Feat_deform) if self.opt.deform else loss_D_GAN_Feat
        _loss_G_VGG = 0.5 * (loss_G_VGG + loss_G_VGG_deform) if self.opt.deform else loss_G_VGG
        _loss_D_real = 0.5 * (loss_D_real + loss_D_real_deform) if self.opt.deform else loss_D_real
        _loss_D_fake = 0.5 * (loss_D_fake + loss_D_fake_deform) if self.opt.deform else loss_D_fake
        _loss_G_GFM = loss_G_GAN_Feat if self.opt.deform and self.opt.gfm else 0
        _loss_edge_penalty = loss_edge_penalty if self.opt.deform and self.opt.use_edge_penalty else 0
        
        # Only return the fake_B image if necessary to save BW
        return_dict['losses'] = self.loss_filter(_loss_G_GAN,
                                                _loss_GAN_Feat,
                                                _loss_G_VGG,
                                                _loss_D_real,
                                                _loss_D_fake,
                                                _loss_G_GFM,
                                                _loss_edge_penalty,
                                                )

        return return_dict
                                    
    def inference(self, sketch, photo=None, sketch_deform=None):
        # Encode Inputs        
        photo = Variable(photo) if photo is not None else None
        sketch_deform = Variable(sketch_deform) if self.opt.deform else None
        sketch, photo, sketch_deform = self.encode_input(Variable(sketch), photo, sketch_deform)
        
        ### compute the complexity of netG
        #flops_netG_sap, params_netG_sap = profile(self.netG_sap, inputs = (sketch_deform,))
        #flops_netG, params_netG = profile(self.netG, inputs = (sketch,))    
        #print('#Params: %.1fG, %.1fM, %.1fG, %.1fM' % (flops_netG, params_netG, flops_netG_sap, params_netG_sap))
        
        return_dict = {} 
        
        if 'dual' in self.opt.netG:
            fake_image_deform, gen_feat_deform = self.netG_sap.forward(sketch_deform)
            fake_image, gen_feat = self.netG_sap.forward(sketch)
            return_dict['fake_image_deform'] = fake_image_deform
            return_dict['fake_image'] = fake_image
            return_dict['gen_feat_deform'] = gen_feat_deform[0]
            return_dict['gen_feat'] = gen_feat[0]
        else:
            fake_image, gen_feat = self.netG.forward(sketch)
            return_dict['fake_image'] = fake_image
        
        return return_dict

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netG_sap, 'G_sap', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG_sap.parameters())
        self.optimizer_G_sap = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst) 