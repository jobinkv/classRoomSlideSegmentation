#!/bin/bash
#	#SBATCH --reservation cvit-trial
#SBATCH --account jobinkv
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=5:00:00
#SBATCH --mail-type=END

module add cuda/9.0
module add cudnn/7-cuda-9.0

model='_model'
#snap='epoch_2_loss_0.02728_acc_0.98999_acc-cls_0.92531_mean-iu_0.88631_fwavacc_0.98043_lr_0.0031325922_model_resnet101_End.pth'
#snap=''
echo "psp on real data ran on: $SLURM_NODELIST"
mkdir -p /ssd_scratch/cvit/jobinkv/data
# geting the image
rsync -avz jobinkv@10.2.16.142:/mnt/1/icdar19/realSlideImgs/cropedSlide/250/img /ssd_scratch/cvit/jobinkv/data/
rsync -avz jobinkv@10.2.16.142:/mnt/1/icdar19/realSlideImgs/cropedSlide/250/ind /ssd_scratch/cvit/jobinkv/data/
rsync -avz jobinkv@10.2.16.142:/mnt/1/icdar19/realSlideImgs/cropedSlide/250/225-25/train.txt /ssd_scratch/cvit/jobinkv/data/
rsync -avz jobinkv@10.2.16.142:/mnt/1/icdar19/realSlideImgs/cropedSlide/250/225-25/val.txt /ssd_scratch/cvit/jobinkv/data/
rsync -avz jobinkv@10.2.16.142:/mnt/1/icdar19/realSlideImgs/cropedSlide/250/225-25/test.txt /ssd_scratch/cvit/jobinkv/data/

# pre-trained models
rsync -avz jobinkv@10.2.16.142:/mnt/1/pyTorchPreTrainedModels /ssd_scratch/cvit/jobinkv/
cd /ssd_scratch/cvit/jobinkv/
mkdir -p $SLURM_JOB_ID$model
#rsync -avz jobinkv@10.2.16.142:/mnt/1/icdar19/trainedmodel/$snap /ssd_scratch/cvit/jobinkv/$SLURM_JOB_ID$model/
#rsync -avz jobinkv@10.2.16.142:/mnt/1/icdar19/trainedmodel/opt_$snap /ssd_scratch/cvit/jobinkv/$SLURM_JOB_ID$model/

# real data for val and test

cd /home/jobinkv/classRoomSlideSegmentation/train/
#  -u USER, --user USER  user id in ada
#  -e EXP, --exp EXP     name of output folder
#  -d DATASET, --dataset DATASET
#                        choose the dataset: cvpr(9 labels) or dsse(7 labels)
#  -n NET, --net NET     choose the network architecture: psp or mfcn
#  -s ,snapshot		 give the trained model for further training
python trainNtestLogs_v7-test.py -u jobinkv -e slideSmal -d slide -n psp -l $SLURM_JOB_ID -m resnet101	# -m resnet152  # -s $snap 
#-m resnet50 #resnet18 
#python trainNtest.py -u jobinkv -e pspOnreal1 -d cvpr -n psp

