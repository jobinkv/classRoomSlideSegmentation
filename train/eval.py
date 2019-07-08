import torch
from PIL import Image
import torchvision.transforms as standard_transforms
import sys
sys.path.insert(0, '../')
from models import *
import ipdb
import numpy as np
model_path = '/ssd_scratch/cvit/jobinkv/slide_trained_on_resnet_101.pth'
# give a slide image here
test_img_path = '/ssd_scratch/cvit/jobinkv/data/img/IMG_20190226_183846_slide.jpg'
def main(model_path,test_img_path):
    net = PSPNet(num_classes=8,resnet = models.resnet101(),res_path='not_required',pretrained=False).cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    mean_std = ([0.9584, 0.9588, 0.9586], [0.1246, 0.1223, 0.1224])

    test_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    img = Image.open(test_img_path).convert('RGB')
    img = test_input_transform(img)
    with torch.no_grad():
        img = img.cuda()
        output = net(img.unsqueeze(0))
        prediction = output.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

if __name__ == '__main__':
    main(model_path,test_img_path)
