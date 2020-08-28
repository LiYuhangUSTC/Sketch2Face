import torch.utils.data as data
from PIL import Image, ImageDraw, ImageChops
import torchvision.transforms as transforms
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    flip = random.random() > 0.5
    
    offset_x=random.randint(-opt.offset_x, opt.offset_x)
    offset_y=random.randint(-opt.offset_y, opt.offset_y)
    degree=random.randint(-opt.degree, opt.degree)    
    
    return {'crop_pos': (x, y), 'flip': flip, 'offset': (offset_x, offset_y), 'degree': degree}

def get_transform(opt, params, method=Image.NEAREST, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    # crop: crop image from large size to target size, 286=>256, 
    # offset: crop image from target size to small size, and then pad back to input size, 256=>246=>256
    # do not apply augmentation for target image (only for sketch)
    if False: #(opt.offset_x != 0 or opt.offset_y != 0):
        transform_list.append(transforms.Lambda(lambda img: __offset(img, params['offset'])))

    if False: # opt.degree != 0:
        transform_list.append(transforms.Lambda(lambda img: __rotate(img, params['degree'])))
        
    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_sketch(opt, params, method=Image.NEAREST, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __resize_sketch(img, opt.loadSize)))
        
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    # crop: crop image from large size to target size, 286=>256, 
    # offset: crop image from target size to small size, and then pad back to input size, 256=>246=>256
    if (opt.offset_x != 0 or opt.offset_y != 0):
        transform_list.append(transforms.Lambda(lambda img: __offset(img, params['offset'])))

    if opt.degree != 0:
        transform_list.append(transforms.Lambda(lambda img: __rotate(img, params['degree'])))
        
    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

    
def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.NEAREST):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __resize_sketch(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    img = img.resize((w, h), Image.BOX)
    box_np = (np.asarray(img)> 200).astype(np.uint8)*255
    return Image.fromarray(box_np)
    
def __scale_width(img, target_width, method=Image.NEAREST):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __rotate(img, degree):
    mode = img.mode
    img = img.convert('RGBA').rotate(degree)
    background = Image.new('RGBA', img.size, (255,) * 4)
    img=Image.composite(img, background, img)
    return img.convert(mode)

def __offset(img, offset):
    x, y = offset
    img = ImageChops.offset(img, x, y)  
    return img
    
def __dilate(image, radius_map):
    pix_image = image.load()         # for quick pixel reading
    pix_map = radius_map.load()     
    width, height = image.size
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for y in range(height):
        for x in range(width):
            if pix_image[x, y] < 125:
                r = pix_map[x, y]
                leftUpPoint = (x-r, y-r)
                rightDownPoint = (x+r, y+r)
                twoPointList = [leftUpPoint, rightDownPoint]
                draw.ellipse(twoPointList, fill=0)
    return output
 
