import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data

num_classes = 8
ignore_label = 255

# for training purpose
#root = '../../../dataset'

# for evaluation on test set

'''
1- Headings:  #FF0000 (red) ok
2- List:      #008D00 (green) ok
3- Paragraph: #5E00FF (blue) ok
4- Figure:    #AC0028 (brown) ok
5- Table:     #F2FF00 (yellow) ok
6- Caption:   #FF47F0 (violet) ok
7- Equation:  #00FFFF (cyan)ok
color map for marmot table detection english dataset
0=background, 1=text, 2=header, 3=figure, 4=list, 5=tablecaption # 6=tablebody, 7=formula, 7=matrix, 8=tablefootnote, 1=footer, 
'''
#palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 0, 0, 128, 128, 128, 0, 0, 128, 128, 128, 0, 128, 128, 128, 128, 64, 0, 0, 10, 200, 180,10,75, 100,150,254] 
#palette = [0, 0, 0,166,206,227,31,120,180 ,178,223,138,51,160,44,251,154,153,227,26,28,253,191,111,255,127,0]
palette = [0,0,0, 255,0,0, 0,255,0, 0,0,255, 255,0,255, 0,255,255, 255,255,0, 0,125,0, 125,0,0]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask
def colorize_mask_combine(mask,img_path):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    org_image = Image.open(os.path.join(img_path)).convert('RGB')
    new_mask.putpalette(palette)
    mask_combine = Image.blend(new_mask.convert("RGB"),org_image,0.5)
    return mask_combine


def make_dataset(mode,root):
    assert mode in ['train', 'val', 'test_eva', 'test','unsuper']
    items = []
    if mode == 'train':
       
        #load page images from marmot table detection dataset english
        img_path = os.path.join(root, 'data', 'img')
        mask_path = os.path.join(root, 'data', 'ind')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'data','train.txt')).readlines()] 
        
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it.split('.')[0] + '.png'))
            items.append(item)
                
    elif mode == 'val':

        #load page images from marmot table detection dataset english
        img_path = os.path.join(root, 'data', 'img')
        mask_path = os.path.join(root, 'data', 'ind')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'data','val.txt')).readlines()] 
        
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it.split('.')[0]+ '.png'))
            items.append(item)

    elif mode == 'test_eva':

        #load page images from marmot table detection dataset english
        img_path = os.path.join(root, 'data', 'img')
        mask_path = os.path.join(root, 'data', 'ind')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'data','test.txt')).readlines()] 
        
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it.split('.')[0]+ '.png'))
            items.append(item)
          
    elif mode == 'unsuper':
        img_path = os.path.join(root, 'data', 'img')
        mask_path = os.path.join(root, 'data', 'ind')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'data','unsuper.txt')).readlines()] 
        
        for it in data_list:
            item = (os.path.join(img_path, it))
            items.append(item)
    else:

        #load page images from marmot table detection dataset english
        img_path = os.path.join(root, 'data', 'img')
        mask_path = os.path.join(root, 'data', 'ind')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'data','test.txt')).readlines()] 
        
        for it in data_list:
            item = (os.path.join(img_path, it))
            items.append(item)
                   
                          
    return items   

 

class DOC(data.Dataset):
    def __init__(self, mode,root, joint_transform=None, transform=None, target_transform=None):
        self.imgs = make_dataset(mode,root)
        if len(self.imgs) == 0:
            raise (RuntimeError('Found 0 images, please check the data set'))
        self.mode = mode
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

       
    def __getitem__(self, index):
        if self.mode == 'test':
            img_path  = self.imgs[index]
            a,b,c,d,e,f,img_name = img_path.split('/')
            img = Image.open(os.path.join(img_path)).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img_name, img
                                  
        if self.mode == 'unsuper':
            img_path  = self.imgs[index]
            a,b,c,d,e,f,img_name = img_path.split('/')
            img = Image.open(os.path.join(img_path)).convert('RGB')


            imgarr = np.array(img)
            imgarr1 = imgarr[:, :, 0]
            mask = Image.fromarray(imgarr1).convert('P')

            if self.joint_transform is not None:
                img, mask = self.joint_transform(img, mask)

            if self.transform is not None:
                img = self.transform(img)
            return img_name, img
                 
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        
        imgarr = np.array(mask)
        imgarr1 = imgarr[:, :, 0]
        mask = Image.fromarray(imgarr1).convert('P')

                
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
            

        if isinstance(img, list) and isinstance(mask, list):
            if self.transform is not None:
                img = [self.transform(e) for e in img]
            if self.target_transform is not None:
                mask = [self.target_transform(e) for e in mask]
            img, mask = torch.stack(img, 0), torch.stack(mask, 0)
            
        else:
            if self.transform is not None:
                img = self.transform(img)
                
            if self.target_transform is not None:
                mask = self.target_transform(mask)
                
        return img, mask

    def __len__(self):
        return len(self.imgs)
        
        
       
    
