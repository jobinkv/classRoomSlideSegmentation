import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from utils import initialize_weights
from utils.misc import Conv2dDeformable



class MFCN(nn.Module):
    def __init__(self, num_classes, pretrained=True, use_aux=False):
        super(MFCN, self).__init__()
        self.use_aux = use_aux
	print "MFCN on progress"
	def create_conv(c1, c2, c3):
		convs = nn.Sequential(
			nn.Conv2d(c1,c2, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(c2, momentum=.95),
			nn.ReLU(inplace=True),
			nn.Conv2d(c2,c3, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(c3, momentum=.95),
			nn.ReLU(inplace=True))
		return convs
	def create_deconv(c1, c2, c3):
		convs = nn.Sequential(
			nn.ConvTranspose2d(c1,c2, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(c2, momentum=.95),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(c2,c3, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(c3, momentum=.95),
			nn.ReLU(inplace=True))
		return convs
	self.layer1 = create_conv(3, 64, 128)	
	self.layer2 = create_conv(128, 128, 128)	
	self.layer3 = create_conv(128, 128, 128)	
	self.layer4 = create_conv(128, 128, 128)
	self.pool1 = nn.MaxPool2d(2, stride=2,return_indices=True)	
	self.pool2 = nn.MaxPool2d(2, stride=2,return_indices=True)	
	self.pool3 = nn.MaxPool2d(2, stride=2,return_indices=True)	
	self.Unpool1 = nn.MaxUnpool2d(2, stride=2)	
	self.Unpool2 = nn.MaxUnpool2d(2, stride=2)	
	self.Unpool3 = nn.MaxUnpool2d(2, stride=2)	
	self.deconv4 = create_deconv(128, 128, 128)
	self.deconv3 = create_deconv(128+128, 128, 128)
	self.deconv2 = create_deconv(128+128, 128, 128)
	self.deconv1 = create_deconv(128+128, 64, num_classes)
	self.deconv1 = self.deconv1[0:5]
        initialize_weights(self.layer1,self.layer2,self.layer3,self.layer4,self.deconv4,self.deconv3,self.deconv2,self.deconv1)
        if use_aux:
            self.aux_logits = nn.Conv2d(128, num_classes, kernel_size=1)
            initialize_weights(self.aux_logits)
    def forward(self, x):
        x_size = x.size()
        fx_1 = self.layer1(x)
        fx_1p,inds1 = self.pool1(fx_1)
        fx_2 = self.layer2(fx_1p)
	del fx_1p
        fx_2p,inds2 = self.pool2(fx_2)
        fx_3 = self.layer3(fx_2p)
	del fx_2p
        fx_3p,inds3 = self.pool3(fx_3)
        fx_4 = self.layer4(fx_3p)
	del fx_3p
        if self.training and self.use_aux:
            aux = self.aux_logits(fx_4)
        de_features4 = self.deconv4(self.Unpool1(fx_4,inds3,output_size=fx_3.size()))
	del fx_4,inds3
        de_features3_1 = self.Unpool2(fx_3,inds2,output_size=fx_2.size())
        de_features3_2 = self.Unpool2(de_features4,inds2,output_size=fx_2.size())
        de_features3 = self.deconv3(torch.cat((de_features3_1,de_features3_2), 1))
	del de_features3_1,de_features3_2,fx_3,inds2	
        de_features2_1 = self.Unpool3(fx_2,inds1,output_size=fx_1.size())
        de_features2_2 = self.Unpool3(de_features3,inds1,output_size=fx_1.size())
        de_features2 = self.deconv2(torch.cat((de_features2_1,de_features2_2), 1))
	del de_features2_1,de_features2_2,inds1,fx_2
        x = self.deconv1(torch.cat((fx_1,de_features2), 1))
        if self.training and self.use_aux:
            return F.upsample(x, x_size[2:], mode='bilinear',align_corners=True), F.upsample(aux, x_size[2:], mode='bilinear',align_corners=True)
        return F.upsample(x, x_size[2:], mode='bilinear',align_corners=True)
        #return de_features1	
	"""
        initialize_weights(self.ppm, self.final)
        """

        
        
        
        
