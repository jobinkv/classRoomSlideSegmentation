import datetime
import math
import os
import random
import tensorboardX
import ipdb
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
import sys
import matplotlib.ticker as ticker
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from subprocess import call
sys.path.insert(0, '../')
from utils import joint_transforms as simul_transforms
from utils import transforms as extended_transforms
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d
from htmlLog import logHtml
cudnn.benchmark = True
import argparse
parser = argparse.ArgumentParser(description='Train document image segmentation')
parser.add_argument('-u','--user', type=str, default='jobinkv',
                    help='user id in ada')
parser.add_argument('-e','--exp', type=str, default='exp1',
                    help='name of output folder')
parser.add_argument('-d','--dataset', type=str, default='slideV1',
                    help='choose the dataset: cvpr(9 labels) or dsse(7 labels)')
parser.add_argument('-n','--net', type=str, default='psp',
                    help='choose the network architecture: psp or mfcn')
parser.add_argument('-s','--snapshot', type=str, default='',
                    help='give the trained model for further training')
parser.add_argument('-l','--log', type=str, default='',
                    help='give the folder name for saving the tensorflow logs')
parser.add_argument('-m','--model', type=str, default='resnet101',
                    help='resnet101,resnet152,resnet18,resnet34,resnet50')
args = parser.parse_args()
print ('The exp arguments are ',args.user,args.exp,args.net,args.dataset)

ckpt_path = '/ssd_scratch/cvit/'+args.user+'/'#input folder
exp_name = args.log #output folder
dataset = args.dataset
jobid=args.log
if dataset=='cvpr':
        from cvpr import doc
elif dataset=='dsse':
        from dsse import doc
elif dataset=='slide':
        from slide import doc
elif dataset=='slideV1':
        from slideV1 import doc
else:
        print ('please specify the dataset') 
network = args.net
snapShort=args.snapshot
if '_' in snapShort :
        args.model = snapShort.split('_')[15]
Dataroot = '/ssd_scratch/cvit/'+args.user #location of data
root1 = '/ssd_scratch/cvit/'+args.user+'/pyTorchPreTrainedModels/'#location of pretrained model
if args.model=='resnet101':
        resnet = models.resnet101()
        res_path = os.path.join(root1,  'resnet101-5d3b4d8f.pth')
if args.model=='resnet152':
        resnet = models.resnet152()
        res_path = os.path.join(root1,  'resnet152-b121ed2d.pth')
if args.model=='resnet18':
        resnet = models.resnet18()
        res_path = os.path.join(root1,  'resnet18-5c106cde.pth')
if args.model=='resnet34':
        resnet = models.resnet34()
        res_path = os.path.join(root1,  'resnet34-333f7ec4.pth')
if args.model=='resnet50':
        resnet = models.resnet50()
        res_path = os.path.join(root1,  'resnet50-19c8e357.pth')
args = {
    'train_batch_size': 8,
    'lr': 5e-3,
    'lr_decay': 0.9,
    'max_iter':300000,  #40e3,
    'input_size': 512,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot':snapShort,  # empty string denotes learning from scratch
    'print_freq': 10,
    'max_epoch':50,
    'dataset':dataset,
    'network':network,
    'jobid':jobid,
    'No_train_images':0,
    'Type_of_train_image':'',
    'Auxilary_loss_contribution':0.5,
    'Pretrained_Model':args.model
}
def ploteIt(tain,aux,val,miou,printFrequncy,fname):
    t = np.arange(1, len(tain)+1, 1)*printFrequncy
    tain=np.asarray(tain)
    fig, ax1 = plt.subplots()
    """
{'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'}
    """
    ax1.set_xlabel('Iteration')
    plt.grid(True)
    ax1.set_ylabel('loss', color='tab:red')
    ax1.plot(t, tain, color='tab:red',label='Train loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    if aux!=[]:
        ax1.plot(t, aux, color='tab:green',label='Auxilary loss')
    ax1.plot(t, val, color='tab:pink',label='validation loss')
    ax1.legend()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Mean IoU', color='tab:blue')  # we already handled the x-label with ax1
    ax2.plot(t, miou, color='tab:blue',label='Mean IoU')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.legend()
    plt.title('Loss Mean IoU curve')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(fname)
    call(["scp", fname+".png", "jobinkv@10.2.16.142:/home/jobinkv/Documents/r1/19wavc/"])
sep_iou_val=[]
sep_iou_test=[]
curr_iter_print=0
def main(train_args):
    print ('No of classes', doc.num_classes)
    if train_args['network']=='psp':
        net = PSPNet(num_classes=doc.num_classes,resnet=resnet,res_path=res_path).cuda()
    elif train_args['network']=='mfcn':
         net = MFCN(num_classes=doc.num_classes,use_aux=True).cuda()
    elif train_args['network']=='psppen':
        net = PSPNet(num_classes=doc.num_classes,resnet=resnet,res_path=res_path).cuda()
    print ("number of cuda devices = ", torch.cuda.device_count())
    if len(train_args['snapshot']) == 0:
        curr_epoch = 1
        train_args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        print ('training resumes from ' + train_args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, 'model_'+exp_name, train_args['snapshot'])))
        split_snapshot = train_args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        train_args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                     'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                     'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}
    net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
    net.train()
    mean_std =([0.9584, 0.9588, 0.9586], [0.1246, 0.1223, 0.1224])
    weight = torch.FloatTensor(doc.num_classes)
    #weight[0] = 1.0/0.5511 # background
    train_simul_transform = simul_transforms.Compose([
        simul_transforms.RandomSized(train_args['input_size']),
        simul_transforms.RandomRotate(3),
        #simul_transforms.RandomHorizontallyFlip()
        simul_transforms.Scale(train_args['input_size']),
        simul_transforms.CenterCrop(train_args['input_size'])
    ])
    val_simul_transform = simul_transforms.Scale(train_args['input_size'])
    train_input_transform = standard_transforms.Compose([
        extended_transforms.RandomGaussianBlur(),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    train_set = doc.DOC('train',Dataroot, joint_transform=train_simul_transform, transform=train_input_transform,
                        target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=train_args['train_batch_size'], num_workers=1, shuffle=True, drop_last = True)
    train_loader_temp = DataLoader(train_set, batch_size=1, num_workers=1, shuffle=True, drop_last = True)
    train_args['No_train_images']=len(train_loader_temp)
    del train_loader_temp
    if train_args['No_train_images']==87677:
        train_args['Type_of_train_image']='All Synthetic slide image'
    elif train_args['No_train_images']==150:
        train_args['Type_of_train_image']='Real image'
    elif train_args['No_train_images']==84641:
        train_args['Type_of_train_image']='Synthetic image'
    elif train_args['No_train_images']==151641:
        train_args['Type_of_train_image']='Real + Synthetic image'
    val_set = doc.DOC('val',Dataroot, joint_transform=val_simul_transform, transform=val_input_transform,
                      target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=1, shuffle=False, drop_last = True)
    #criterion = CrossEntropyLoss2d(weight = weight, size_average = True, ignore_index = doc.ignore_label).cuda()
    criterion = CrossEntropyLoss2d(size_average=True, ignore_index=doc.ignore_label).cuda()
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': train_args['weight_decay']}
    ], momentum=train_args['momentum'])
    if len(train_args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, 'model_'+exp_name, 'opt_' + train_args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * train_args['lr']
        optimizer.param_groups[1]['lr'] = train_args['lr']
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(train_args) + '\n\n')
    train(train_loader, net, criterion, optimizer, curr_epoch, train_args, val_loader)
    #train(train_set, net, criterion, optimizer, curr_epoch, train_args, val_loader)

def train(train_loader, net, criterion, optimizer, curr_epoch, train_args, val_loader):
    print('===:Training Starts:===')
    plot_losses_train=[]
    plot_losses_val=[]
    plot_losses_aux=[]
    plot_mean_iu=[]
    while True:
        train_main_loss = AverageMeter()
        #if train_args['network']=='psp':
        train_aux_loss = AverageMeter()
        curr_iter = (curr_epoch - 1) * len(train_loader)
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                      ) ** train_args['lr_decay']
            optimizer.param_groups[1]['lr'] = train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                  ) ** train_args['lr_decay']
            inputs, labels = data
            assert inputs.size()[2:] == labels.size()[1:]
            N = inputs.size(0) * inputs.size(2) * inputs.size(3)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            #if train_args['network']=='psp':
            outputs, aux = net(inputs)
            #elif train_args['network']=='mfcn':
            #   outputs = net(inputs)

            assert outputs.size()[2:] == labels.size()[1:]
            assert outputs.size()[1] == doc.num_classes

            main_loss = criterion(outputs, labels)
            #if train_args['network']=='psp':
            aux_loss = criterion(aux, labels)
            loss = main_loss + train_args['Auxilary_loss_contribution'] * aux_loss
            #elif train_args['network']=='mfcn':
            #   loss = main_loss

            loss.backward()
            optimizer.step()

            train_main_loss.update(main_loss.item(), N)
            #if train_args['network']=='psp':
            train_aux_loss.update(aux_loss.item(), N)

            curr_iter += 1
            if (i + 1) % train_args['print_freq'] == 0:
                #if train_args['network']=='psp':
                print ('Train:  [psp epoch %d], [iter %d/%d],[train main loss %.5f], [train aux loss %.5f],[total loss %.5f],[lr %.10f] on %s' % (curr_epoch, i + 1, len(train_loader), train_main_loss.avg, train_aux_loss.avg,loss,
                        optimizer.param_groups[1]['lr'],train_args['dataset']
                        ))
                plot_losses_aux.append(train_aux_loss.avg)
                #elif train_args['network']=='mfcn':
                #       print 'Train: [mfcn epoch %d], [iter %d / %d], [train main loss %.5f], [total loss %.5f], [lr %.10f] on %s' % (
                #       curr_epoch, i + 1, len(train_loader), train_main_loss.avg,loss,
                #       optimizer.param_groups[1]['lr'],train_args['dataset']
                #       )
                plot_losses_train.append(train_main_loss.avg)
                val_loss_val,mean_iu_val = validate(val_loader, net, criterion, optimizer, curr_epoch, train_args,curr_iter)
                plot_mean_iu.append(mean_iu_val)
                plot_losses_val.append(val_loss_val)
                ploteIt(plot_losses_train,plot_losses_aux,plot_losses_val,
                        plot_mean_iu,train_args['print_freq'],train_args['jobid'])
                logHtml(sep_iou_val,sep_iou_test,train_args)
                #showPlot(plot_losses_train,plot_losses_val,plot_losses_aux,plot_mean_iu,
                #       fileN='Error_loss_curve_'+train_args['jobid'], caption=train_args['network']+' on '+train_args['dataset'])
            if curr_iter >= train_args['max_iter']:
                return
        #validate(val_loader, net, criterion, optimizer, curr_epoch, train_args)
        curr_epoch += 1

def validate(val_loader, net, criterion, optimizer, epoch, train_args,curr_iter):
    #print('===:Validate Trained Model:===')
    net.eval()
    val_loss = AverageMeter()
    gts_all, predictions_all = [], []
    for vi, data in enumerate(val_loader):
        inputs, gts = data
        N = inputs.size(0) * inputs.size(2) * inputs.size(3)
        with torch.no_grad():
                inputs = inputs.cuda()
                gts = gts.cuda()
                outputs = net(inputs)
                predictions = outputs.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
                val_loss.update(criterion(outputs, gts).cpu().numpy(), N)
                tempss = gts.squeeze_(0).cpu().numpy()
                gts_all.append(tempss)
                predictions_all.append(predictions)
    acc, acc_cls, mean_iu, fwavacc, sep_iu = evaluate(predictions_all, gts_all, doc.num_classes)
    del predictions_all

    if mean_iu > 0.4: # train_args['best_record']['mean_iu']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc

        snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr']
        )
        traiModelPath = os.path.join(ckpt_path, 'model_'+exp_name, snapshot_name + '.pth')
        trainModel_opt = os.path.join(ckpt_path, 'model_'+exp_name, 'opt_' + snapshot_name + '.pth')
        check_mkdir(os.path.join(ckpt_path, 'model_'+exp_name))
        torch.save(net.module.state_dict(), traiModelPath)
        #torch.save(optimizer.state_dict(), trainModel_opt)
        call(["scp", traiModelPath, "jobinkv@10.2.16.142:/mnt/1/icdar19/trainedmodel/fine/"])
        call(["scp", trainModel_opt, "jobinkv@10.2.16.142:/mnt/1/icdar19/trainedmodel/fine/"])
        print (snapshot_name + '.pth')
        print ('opt_' + snapshot_name + '.pth')
        testing(net,train_args,curr_iter)
    sep_iu = sep_iu.tolist()
    sep_iu.append(mean_iu)
    sep_iu.insert(0,curr_iter)
    sep_iou_val.append(sep_iu)
    #print '--------------------------------------------------------------------'
    print ('Current: [epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f],[%s]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc,train_args['network']))

    print ('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d],[%s]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch'],train_args['network']))
    #print '--------------------------------------------------------------------'
    net.train()
    return val_loss.avg,mean_iu
def testing(net,train_args,curr_iter):
    net.eval()

    mean_std = ([0.9584, 0.9588, 0.9586], [0.1246, 0.1223, 0.1224])

    test_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    #target_transform = extended_transforms.MaskToTensor()
    scaleTest_transform = simul_transforms.Scale(512)
    test_simul_transform = simul_transforms.Scale(train_args['input_size'])
    #val_input_transform = standard_transforms.Compose([
    #    standard_transforms.ToTensor(),
    #    standard_transforms.Normalize(*mean_std)
    #test_set = doc.DOC('val',Dataroot, transform=val_input_transform,
    #                 target_transform=target_transform)

    # segmentation on test images
    test_set = doc.DOC('test',ckpt_path,joint_transform=test_simul_transform,transform=test_input_transform)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=1, shuffle=False)

    check_mkdir(os.path.join(ckpt_path, exp_name,str(curr_iter)))
    listOfImgs=['2018_101008181.jpg','2018_101001826.jpg','2016_101000408.jpg','2009_052067571.jpg','2010_055401143.jpg']
    for vi, data in enumerate(test_loader):
        img_name, img = data
        img_name = img_name[0]
        with torch.no_grad():
                img = img.cuda()
                output = net(img)

                prediction = output.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
                imput_img = img.squeeze_(0).cpu().numpy()
                prediction = doc.colorize_mask_combine(prediction,ckpt_path+'data/img/'+img_name)
                #if img_name in listOfImgs:
                if curr_iter > 1:
                        prediction.save(os.path.join(ckpt_path, exp_name,str(curr_iter), img_name ))
    call(["rsync","-avz", os.path.join(ckpt_path, exp_name), "jobinkv@10.2.16.142:/home/jobinkv/Documents/r1/19wavc/"])
                #print '%d / %d' % (vi + 1, len(test_loader))

    test_set_eva = doc.DOC('test_eva',ckpt_path, joint_transform=test_simul_transform, transform=test_input_transform,target_transform=target_transform)
    test_loader_eva = DataLoader(test_set_eva, batch_size=1, num_workers=1, shuffle=False)

    gts_all, predictions_all = [], []

    for vi, data in enumerate(test_loader_eva):
        img, gts = data
        with torch.no_grad():
                img = img.cuda()
                gts = gts.cuda()

                output = net(img)
                prediction = output.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

                gts_all.append(gts.squeeze_(0).cpu().numpy())
                predictions_all.append(prediction)

    acc, acc_cls, mean_iu, fwavacc, sep_iu = evaluate(predictions_all, gts_all, doc.num_classes)
    del predictions_all
    print ('--------------------------------------------------------------------')
    print ('[test acc %.5f], [test acc_cls %.5f], [test mean_iu %.5f], [test fwavacc %.5f]' % (acc, acc_cls, mean_iu, fwavacc))
    print ('--------------------------------------------------------------------')
    sep_iu = sep_iu.tolist()
    sep_iu.append(mean_iu)
    sep_iu.insert(0,curr_iter)
    sep_iou_test.append(sep_iu)

if __name__ == '__main__':
    main(args)
