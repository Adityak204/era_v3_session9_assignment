# Imagenet-1k Classification using ResNet50
This repository contains the code for training and evaluating ResNet50 on Imagenet-1k dataset.
Model used here is ResNet50 model from PyTorch without pretrained weights.
For training have made use of EC2 instance with GPU support on AWS.
Imagenet-1k dataset has 1,281,167 training images and 50,000 validation images with 1000 labeled classes.

## Model running on HuggingFace
Model is available on HuggingFace model hub. You test the model here: https://huggingface.co/spaces/Adityak204/ResNetVision-1K
<br>
Screenshot from huggingface: ![image](data/hf-screenshot.png)

## Data Source
The Imagenet-1k dataset was downloaded from kaggle. You can download the dataset from 
- Training : https://www.kaggle.com/c/imagenet-object-localization-challenge/data
- Validation : https://www.kaggle.com/datasets/tusonggao/imagenet-validation-dataset

## Process Highlight
- Created EBS (gp3) volume of 400 GB (as dataset size is ~160 GB zipped version) and attached to EC2 instance.
    - Steps followed:
        - Need to make sure that the instance and EBS are in same zone (Ex: east-us-1a)
        - Switch on the instance and go to EC2 Dashboard -> Elast Block Store -> Volumes -> Right click on the volume -> Attach Volume -> Attach to the instance -> Device name: /dev/sdf
        - SSH into the instance and run `lsblk` to check the attached volume.
        - Run `sudo file -s /dev/xvdf` to check the file system of the volume.
        - If not files system found -> Run `sudo mkfs -t ext4 /dev/xvdf` to format the volume.
        - Create a directory to mount the volume: `mkdir -p /home/ec2-user/ebs/volumes`
        - Mount the volume: `sudo mount /dev/xvdf /home/ec2-user/ebs/volumes`

- Made used of Kaggle api to download the dataset on EC2 instance.
    - Install Kaggle api: `pip install kaggle`
    - Create Kaggle API token and save it in `~/.config/kaggle/kaggle.json`
    - Download dataset: `kaggle competitions download -c imagenet-object-localization-challenge` 
    - Unzip the data set (this will take sometime)

- Model training and evaluation
    - Made use of Pytorch Lightning for training and evaluation.
    - Model was trained on g6.12xlarge machine containing 4 - L4 GPUs
    - Pytorch Lightning simplified the process of DDP and mixed precision training
    - Model was trained for 36 epochs @ 16-17mins per epochs
    - Training and Validation logs and model checkpoints were save on EBS

## Model Highlight and Results
- Model used is ResNet50 from PyTorch without pretrained weights
- Optimizer used: SGD with momentum of 0.9 and weight decay of 1e-4
- One Cycle LR scheduler was used with max_lr=0.15
- Model was evaluated on validation dataset and achieved top-1 accuracy of 60.35%
- As credits were expiring on AWS, training was stopped after 36 epochs. 
- Model can be further trained to improve the accuracy. 
- (Last epoch LR: 0.137491,  annelaing process was yet to start)

## Training logs
```
2024-12-30 19:26:57 | INFO | Logging setup complete. Logs will be saved to: /home/ec2-user/ebs/volumes/era_session9/training_20241230_192657.log
2024-12-30 19:26:57 | INFO | Starting training with configuration:
2024-12-30 19:26:57 | INFO | PyTorch version: 2.5.1
2024-12-30 19:26:57 | INFO | Logging setup complete. Logs will be saved to: /home/ec2-user/ebs/volumes/era_session9/training_20241230_192657.log
2024-12-30 19:26:57 | INFO | Logging setup complete. Logs will be saved to: /home/ec2-user/ebs/volumes/era_session9/training_20241230_192657.log
2024-12-30 19:26:57 | INFO | Starting training with configuration:
2024-12-30 19:26:57 | INFO | Starting training with configuration:
2024-12-30 19:26:57 | INFO | PyTorch version: 2.5.1
2024-12-30 19:26:57 | INFO | PyTorch version: 2.5.1
2024-12-30 19:26:57 | INFO | CUDA available: True
2024-12-30 19:26:57 | INFO | CUDA device count: 4
2024-12-30 19:26:57 | INFO | CUDA available: True
2024-12-30 19:26:57 | INFO | CUDA device count: 4
2024-12-30 19:26:57 | INFO | CUDA available: True
2024-12-30 19:26:57 | INFO | CUDA device count: 4
2024-12-30 19:26:57 | INFO | CUDA devices: ['NVIDIA L4', 'NVIDIA L4', 'NVIDIA L4', 'NVIDIA L4']
2024-12-30 19:26:57 | INFO | CUDA devices: ['NVIDIA L4', 'NVIDIA L4', 'NVIDIA L4', 'NVIDIA L4']
2024-12-30 19:26:57 | INFO | CUDA devices: ['NVIDIA L4', 'NVIDIA L4', 'NVIDIA L4', 'NVIDIA L4']
2024-12-30 19:26:57 | INFO | Model configuration:
2024-12-30 19:26:57 | INFO | Learning rate: 0.156
2024-12-30 19:26:57 | INFO | Batch size: 256
2024-12-30 19:26:57 | INFO | Number of workers: 16
2024-12-30 19:26:57 | INFO | Max epochs: 40
2024-12-30 19:26:57 | INFO | Starting training
2024-12-30 19:26:57 | INFO | Model configuration:
2024-12-30 19:26:57 | INFO | Learning rate: 0.156
2024-12-30 19:26:57 | INFO | Batch size: 256
2024-12-30 19:26:57 | INFO | Number of workers: 16
2024-12-30 19:26:57 | INFO | Max epochs: 40
2024-12-30 19:26:57 | INFO | Model configuration:
2024-12-30 19:26:57 | INFO | Learning rate: 0.156
2024-12-30 19:26:57 | INFO | Batch size: 256
2024-12-30 19:26:57 | INFO | Number of workers: 16
2024-12-30 19:26:57 | INFO | Max epochs: 40
2024-12-30 19:26:57 | INFO | Starting training
2024-12-30 19:26:57 | INFO | Starting training
2024-12-30 19:27:09 | INFO | Validation metrics - Loss: 52.9518, Accuracy: 0.0000
2024-12-30 19:27:09 | INFO | Validation metrics - Loss: 52.6007, Accuracy: 0.0000
2024-12-30 19:27:09 | INFO | Validation metrics - Loss: 52.4139, Accuracy: 0.0000
2024-12-30 19:27:14 | INFO | 
==================== Epoch 0 ====================
2024-12-30 19:27:14 | INFO | 
==================== Epoch 0 ====================
2024-12-30 19:27:14 | INFO | 
==================== Epoch 0 ====================
2024-12-30 19:43:24 | INFO | New best validation accuracy: 4.5508. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch00-acc4.5508.ckpt
2024-12-30 19:43:24 | INFO | New best validation accuracy: 4.6418. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch00-acc4.6418.ckpt
2024-12-30 19:43:24 | INFO | New best validation accuracy: 4.4458. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch00-acc4.4458.ckpt
2024-12-30 19:43:24 | INFO | Validation metrics - Loss: 5.6285, Accuracy: 4.5508
2024-12-30 19:43:24 | INFO | Validation metrics - Loss: 5.6539, Accuracy: 4.4458
2024-12-30 19:43:24 | INFO | Validation metrics - Loss: 5.6396, Accuracy: 4.6418
2024-12-30 19:43:24 | INFO | Training metrics - Loss: 6.3321, Accuracy: 1.7064, LR: 0.006401
2024-12-30 19:43:24 | INFO | Training metrics - Loss: 6.3311, Accuracy: 1.7248, LR: 0.006401
2024-12-30 19:43:24 | INFO | Training metrics - Loss: 6.3305, Accuracy: 1.6954, LR: 0.006401
2024-12-30 19:43:24 | INFO | 
==================== Epoch 1 ====================
2024-12-30 19:43:24 | INFO | 
==================== Epoch 1 ====================
2024-12-30 19:43:24 | INFO | 
==================== Epoch 1 ====================
2024-12-30 19:58:52 | INFO | New best validation accuracy: 9.5322. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch01-acc9.5322.ckpt
2024-12-30 19:58:52 | INFO | New best validation accuracy: 9.5429. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch01-acc9.5429.ckpt
2024-12-30 19:58:52 | INFO | New best validation accuracy: 9.0214. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch01-acc9.0214.ckpt
2024-12-30 19:58:52 | INFO | Validation metrics - Loss: 4.9919, Accuracy: 9.5429
2024-12-30 19:58:52 | INFO | Validation metrics - Loss: 4.9676, Accuracy: 9.5322
2024-12-30 19:58:52 | INFO | Validation metrics - Loss: 5.0144, Accuracy: 9.0214
2024-12-30 19:58:52 | INFO | Training metrics - Loss: 5.3372, Accuracy: 6.8533, LR: 0.006881
2024-12-30 19:58:52 | INFO | Training metrics - Loss: 5.3355, Accuracy: 6.8170, LR: 0.006881
2024-12-30 19:58:52 | INFO | Training metrics - Loss: 5.3333, Accuracy: 6.8002, LR: 0.006881
2024-12-30 19:58:52 | INFO | 
==================== Epoch 2 ====================
2024-12-30 19:58:52 | INFO | 
==================== Epoch 2 ====================
2024-12-30 19:58:52 | INFO | 
==================== Epoch 2 ====================
2024-12-30 20:14:32 | INFO | New best validation accuracy: 17.3575. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch02-acc17.3575.ckpt
2024-12-30 20:14:32 | INFO | New best validation accuracy: 17.9348. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch02-acc17.9348.ckpt
2024-12-30 20:14:32 | INFO | New best validation accuracy: 17.1073. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch02-acc17.1073.ckpt
2024-12-30 20:14:32 | INFO | Validation metrics - Loss: 4.1820, Accuracy: 17.9348
2024-12-30 20:14:32 | INFO | Validation metrics - Loss: 4.2421, Accuracy: 17.1073
2024-12-30 20:14:32 | INFO | Validation metrics - Loss: 4.1911, Accuracy: 17.3575
2024-12-30 20:14:32 | INFO | Training metrics - Loss: 4.7207, Accuracy: 12.6534, LR: 0.007681
2024-12-30 20:14:32 | INFO | Training metrics - Loss: 4.7214, Accuracy: 12.5532, LR: 0.007681
2024-12-30 20:14:32 | INFO | Training metrics - Loss: 4.7175, Accuracy: 12.6216, LR: 0.007681
2024-12-30 20:14:33 | INFO | 
==================== Epoch 3 ====================
2024-12-30 20:14:33 | INFO | 
==================== Epoch 3 ====================
2024-12-30 20:14:33 | INFO | 
==================== Epoch 3 ====================
2024-12-30 20:30:01 | INFO | New best validation accuracy: 22.5451. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch03-acc22.5451.ckpt
2024-12-30 20:30:01 | INFO | New best validation accuracy: 22.9632. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch03-acc22.9632.ckpt
2024-12-30 20:30:01 | INFO | Validation metrics - Loss: 3.7941, Accuracy: 22.5451
2024-12-30 20:30:01 | INFO | Validation metrics - Loss: 3.8007, Accuracy: 22.9632
2024-12-30 20:30:01 | INFO | Training metrics - Loss: 4.2176, Accuracy: 18.5260, LR: 0.008795
2024-12-30 20:30:01 | INFO | Training metrics - Loss: 4.2233, Accuracy: 18.4396, LR: 0.008795
2024-12-30 20:30:01 | INFO | New best validation accuracy: 22.8115. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch03-acc22.8115.ckpt
2024-12-30 20:30:01 | INFO | Validation metrics - Loss: 3.8305, Accuracy: 22.8115
2024-12-30 20:30:01 | INFO | Training metrics - Loss: 4.2152, Accuracy: 18.5602, LR: 0.008795
2024-12-30 20:30:01 | INFO | 
==================== Epoch 4 ====================
2024-12-30 20:30:01 | INFO | 
==================== Epoch 4 ====================
2024-12-30 20:30:01 | INFO | 
==================== Epoch 4 ====================
2024-12-30 20:45:29 | INFO | New best validation accuracy: 27.7445. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch04-acc27.7445.ckpt
2024-12-30 20:45:29 | INFO | New best validation accuracy: 27.3926. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch04-acc27.3926.ckpt
2024-12-30 20:45:29 | INFO | New best validation accuracy: 26.6245. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch04-acc26.6245.ckpt
2024-12-30 20:45:29 | INFO | Validation metrics - Loss: 3.4636, Accuracy: 27.7445
2024-12-30 20:45:29 | INFO | Validation metrics - Loss: 3.4818, Accuracy: 27.3926
2024-12-30 20:45:29 | INFO | Validation metrics - Loss: 3.5281, Accuracy: 26.6245
2024-12-30 20:45:29 | INFO | Training metrics - Loss: 3.8273, Accuracy: 23.7998, LR: 0.010219
2024-12-30 20:45:29 | INFO | Training metrics - Loss: 3.8282, Accuracy: 23.7493, LR: 0.010219
2024-12-30 20:45:29 | INFO | Training metrics - Loss: 3.8281, Accuracy: 23.8476, LR: 0.010219
2024-12-30 20:45:30 | INFO | 
==================== Epoch 5 ====================
2024-12-30 20:45:30 | INFO | 
==================== Epoch 5 ====================
2024-12-30 20:45:30 | INFO | 
==================== Epoch 5 ====================
2024-12-30 21:00:58 | INFO | New best validation accuracy: 31.7834. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch05-acc31.7834.ckpt
2024-12-30 21:00:58 | INFO | New best validation accuracy: 32.3557. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch05-acc32.3557.ckpt
2024-12-30 21:00:58 | INFO | New best validation accuracy: 31.7331. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch05-acc31.7331.ckpt
2024-12-30 21:00:58 | INFO | Validation metrics - Loss: 3.1675, Accuracy: 32.3557
2024-12-30 21:00:58 | INFO | Validation metrics - Loss: 3.2230, Accuracy: 31.7834
2024-12-30 21:00:58 | INFO | Validation metrics - Loss: 3.1644, Accuracy: 31.7331
2024-12-30 21:00:58 | INFO | Training metrics - Loss: 3.5233, Accuracy: 28.1059, LR: 0.011947
2024-12-30 21:00:58 | INFO | Training metrics - Loss: 3.5227, Accuracy: 28.1496, LR: 0.011947
2024-12-30 21:00:58 | INFO | Training metrics - Loss: 3.5202, Accuracy: 28.2621, LR: 0.011947
2024-12-30 21:00:58 | INFO | 
==================== Epoch 6 ====================
2024-12-30 21:00:58 | INFO | 
==================== Epoch 6 ====================
2024-12-30 21:00:58 | INFO | 
==================== Epoch 6 ====================
2024-12-30 21:16:29 | INFO | New best validation accuracy: 36.4904. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch06-acc36.4904.ckpt
2024-12-30 21:16:29 | INFO | New best validation accuracy: 37.9031. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch06-acc37.9031.ckpt
2024-12-30 21:16:29 | INFO | New best validation accuracy: 37.4776. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch06-acc37.4776.ckpt
2024-12-30 21:16:29 | INFO | Validation metrics - Loss: 2.9136, Accuracy: 36.4904
2024-12-30 21:16:29 | INFO | Validation metrics - Loss: 2.8416, Accuracy: 37.9031
2024-12-30 21:16:29 | INFO | Validation metrics - Loss: 2.8398, Accuracy: 37.4776
2024-12-30 21:16:29 | INFO | Training metrics - Loss: 3.2749, Accuracy: 32.0788, LR: 0.013972
2024-12-30 21:16:29 | INFO | Training metrics - Loss: 3.2717, Accuracy: 32.1334, LR: 0.013972
2024-12-30 21:16:29 | INFO | Training metrics - Loss: 3.2726, Accuracy: 32.0726, LR: 0.013972
2024-12-30 21:16:30 | INFO | 
==================== Epoch 7 ====================
2024-12-30 21:16:30 | INFO | 
==================== Epoch 7 ====================
2024-12-30 21:16:30 | INFO | 
==================== Epoch 7 ====================
2024-12-30 21:31:56 | INFO | New best validation accuracy: 39.3823. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch07-acc39.3823.ckpt
2024-12-30 21:31:56 | INFO | New best validation accuracy: 39.9027. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch07-acc39.9027.ckpt
2024-12-30 21:31:56 | INFO | New best validation accuracy: 40.5808. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch07-acc40.5808.ckpt
2024-12-30 21:31:56 | INFO | Validation metrics - Loss: 2.7658, Accuracy: 39.3823
2024-12-30 21:31:56 | INFO | Validation metrics - Loss: 2.6917, Accuracy: 40.5808
2024-12-30 21:31:56 | INFO | Validation metrics - Loss: 2.6995, Accuracy: 39.9027
2024-12-30 21:31:56 | INFO | Training metrics - Loss: 3.0757, Accuracy: 35.3177, LR: 0.016284
2024-12-30 21:31:56 | INFO | Training metrics - Loss: 3.0687, Accuracy: 35.3929, LR: 0.016284
2024-12-30 21:31:56 | INFO | Training metrics - Loss: 3.0728, Accuracy: 35.4350, LR: 0.016284
2024-12-30 21:31:57 | INFO | 
==================== Epoch 8 ====================
2024-12-30 21:31:57 | INFO | 
==================== Epoch 8 ====================
2024-12-30 21:31:57 | INFO | 
==================== Epoch 8 ====================
2024-12-30 21:47:25 | INFO | New best validation accuracy: 40.3093. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch08-acc40.3093.ckpt
2024-12-30 21:47:25 | INFO | New best validation accuracy: 40.9759. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch08-acc40.9759.ckpt
2024-12-30 21:47:25 | INFO | New best validation accuracy: 41.3357. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch08-acc41.3357.ckpt
2024-12-30 21:47:25 | INFO | Validation metrics - Loss: 2.7121, Accuracy: 40.3093
2024-12-30 21:47:25 | INFO | Validation metrics - Loss: 2.6610, Accuracy: 40.9759
2024-12-30 21:47:25 | INFO | Validation metrics - Loss: 2.6665, Accuracy: 41.3357
2024-12-30 21:47:25 | INFO | Training metrics - Loss: 2.8997, Accuracy: 38.2646, LR: 0.018875
2024-12-30 21:47:25 | INFO | Training metrics - Loss: 2.9044, Accuracy: 38.1629, LR: 0.018875
2024-12-30 21:47:25 | INFO | Training metrics - Loss: 2.9029, Accuracy: 38.3326, LR: 0.018875
2024-12-30 21:47:26 | INFO | 
==================== Epoch 9 ====================
2024-12-30 21:47:26 | INFO | 
==================== Epoch 9 ====================
2024-12-30 21:47:26 | INFO | 
==================== Epoch 9 ====================
2024-12-30 22:02:52 | INFO | New best validation accuracy: 42.7570. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch09-acc42.7570.ckpt
2024-12-30 22:02:52 | INFO | New best validation accuracy: 41.8067. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch09-acc41.8067.ckpt
2024-12-30 22:02:52 | INFO | New best validation accuracy: 41.8817. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch09-acc41.8817.ckpt
2024-12-30 22:02:52 | INFO | Validation metrics - Loss: 2.5891, Accuracy: 42.7570
2024-12-30 22:02:52 | INFO | Validation metrics - Loss: 2.6601, Accuracy: 41.8067
2024-12-30 22:02:52 | INFO | Validation metrics - Loss: 2.6162, Accuracy: 41.8817
2024-12-30 22:02:52 | INFO | Training metrics - Loss: 2.7594, Accuracy: 40.5229, LR: 0.021732
2024-12-30 22:02:52 | INFO | Training metrics - Loss: 2.7578, Accuracy: 40.6303, LR: 0.021732
2024-12-30 22:02:52 | INFO | Training metrics - Loss: 2.7640, Accuracy: 40.4849, LR: 0.021732
2024-12-30 22:02:52 | INFO | 
==================== Epoch 10 ====================
2024-12-30 22:02:52 | INFO | 
==================== Epoch 10 ====================
2024-12-30 22:02:52 | INFO | 
==================== Epoch 10 ====================
2024-12-30 22:18:20 | INFO | New best validation accuracy: 45.5601. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch10-acc45.5601.ckpt
2024-12-30 22:18:20 | INFO | New best validation accuracy: 46.2902. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch10-acc46.2902.ckpt
2024-12-30 22:18:20 | INFO | New best validation accuracy: 45.7531. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch10-acc45.7531.ckpt
2024-12-30 22:18:20 | INFO | Validation metrics - Loss: 2.3715, Accuracy: 45.5601
2024-12-30 22:18:20 | INFO | Validation metrics - Loss: 2.4297, Accuracy: 45.7531
2024-12-30 22:18:20 | INFO | Validation metrics - Loss: 2.3615, Accuracy: 46.2902
2024-12-30 22:18:20 | INFO | Training metrics - Loss: 2.6363, Accuracy: 42.7387, LR: 0.024844
2024-12-30 22:18:20 | INFO | Training metrics - Loss: 2.6445, Accuracy: 42.6840, LR: 0.024844
2024-12-30 22:18:20 | INFO | Training metrics - Loss: 2.6442, Accuracy: 42.5985, LR: 0.024844
2024-12-30 22:18:21 | INFO | 
==================== Epoch 11 ====================
2024-12-30 22:18:21 | INFO | 
==================== Epoch 11 ====================
2024-12-30 22:18:21 | INFO | 
==================== Epoch 11 ====================
2024-12-30 22:33:47 | INFO | New best validation accuracy: 47.5644. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch11-acc47.5644.ckpt
2024-12-30 22:33:47 | INFO | New best validation accuracy: 46.4735. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch11-acc46.4735.ckpt
2024-12-30 22:33:47 | INFO | New best validation accuracy: 47.1648. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch11-acc47.1648.ckpt
2024-12-30 22:33:47 | INFO | Validation metrics - Loss: 2.3404, Accuracy: 47.5644
2024-12-30 22:33:47 | INFO | Validation metrics - Loss: 2.4088, Accuracy: 46.4735
2024-12-30 22:33:47 | INFO | Validation metrics - Loss: 2.3446, Accuracy: 47.1648
2024-12-30 22:33:47 | INFO | Training metrics - Loss: 2.5398, Accuracy: 44.4179, LR: 0.028197
2024-12-30 22:33:47 | INFO | Training metrics - Loss: 2.5452, Accuracy: 44.3277, LR: 0.028197
2024-12-30 22:33:47 | INFO | Training metrics - Loss: 2.5447, Accuracy: 44.3492, LR: 0.028197
2024-12-30 22:33:48 | INFO | 
==================== Epoch 12 ====================
2024-12-30 22:33:48 | INFO | 
==================== Epoch 12 ====================
2024-12-30 22:33:48 | INFO | 
==================== Epoch 12 ====================
2024-12-30 22:49:16 | INFO | New best validation accuracy: 48.7877. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch12-acc48.7877.ckpt
2024-12-30 22:49:16 | INFO | New best validation accuracy: 48.5383. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch12-acc48.5383.ckpt
2024-12-30 22:49:16 | INFO | New best validation accuracy: 49.0216. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch12-acc49.0216.ckpt
2024-12-30 22:49:16 | INFO | Validation metrics - Loss: 2.2712, Accuracy: 48.7877
2024-12-30 22:49:16 | INFO | Validation metrics - Loss: 2.2045, Accuracy: 48.5383
2024-12-30 22:49:16 | INFO | Validation metrics - Loss: 2.2023, Accuracy: 49.0216
2024-12-30 22:49:16 | INFO | Training metrics - Loss: 2.4485, Accuracy: 46.0768, LR: 0.031778
2024-12-30 22:49:16 | INFO | Training metrics - Loss: 2.4479, Accuracy: 46.0996, LR: 0.031778
2024-12-30 22:49:16 | INFO | Training metrics - Loss: 2.4564, Accuracy: 46.0192, LR: 0.031778
2024-12-30 22:49:16 | INFO | 
==================== Epoch 13 ====================
2024-12-30 22:49:16 | INFO | 
==================== Epoch 13 ====================
2024-12-30 22:49:16 | INFO | 
==================== Epoch 13 ====================
2024-12-30 23:04:44 | INFO | Validation metrics - Loss: 2.4169, Accuracy: 46.2377
2024-12-30 23:04:44 | INFO | Training metrics - Loss: 2.3819, Accuracy: 47.2710, LR: 0.035569
2024-12-30 23:04:44 | INFO | Validation metrics - Loss: 2.4692, Accuracy: 45.9790
2024-12-30 23:04:44 | INFO | Training metrics - Loss: 2.3796, Accuracy: 47.2651, LR: 0.035569
2024-12-30 23:04:44 | INFO | Validation metrics - Loss: 2.3936, Accuracy: 46.4414
2024-12-30 23:04:44 | INFO | Training metrics - Loss: 2.3774, Accuracy: 47.3615, LR: 0.035569
2024-12-30 23:04:44 | INFO | 
==================== Epoch 14 ====================
2024-12-30 23:04:44 | INFO | 
==================== Epoch 14 ====================
2024-12-30 23:04:44 | INFO | 
==================== Epoch 14 ====================
2024-12-30 23:20:11 | INFO | Validation metrics - Loss: 2.3217, Accuracy: 47.8444
2024-12-30 23:20:11 | INFO | Training metrics - Loss: 2.3088, Accuracy: 48.6993, LR: 0.039557
2024-12-30 23:20:12 | INFO | Validation metrics - Loss: 2.2623, Accuracy: 48.0610
2024-12-30 23:20:12 | INFO | Training metrics - Loss: 2.3076, Accuracy: 48.6053, LR: 0.039557
2024-12-30 23:20:12 | INFO | Validation metrics - Loss: 2.2733, Accuracy: 48.2680
2024-12-30 23:20:12 | INFO | Training metrics - Loss: 2.3158, Accuracy: 48.4650, LR: 0.039557
2024-12-30 23:20:12 | INFO | 
==================== Epoch 15 ====================
2024-12-30 23:20:12 | INFO | 
==================== Epoch 15 ====================
2024-12-30 23:20:12 | INFO | 
==================== Epoch 15 ====================
2024-12-30 23:35:39 | INFO | New best validation accuracy: 50.0569. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch15-acc50.0569.ckpt
2024-12-30 23:35:39 | INFO | New best validation accuracy: 50.2399. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch15-acc50.2399.ckpt
2024-12-30 23:35:39 | INFO | New best validation accuracy: 50.2748. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch15-acc50.2748.ckpt
2024-12-30 23:35:39 | INFO | Validation metrics - Loss: 2.2116, Accuracy: 50.2399
2024-12-30 23:35:39 | INFO | Validation metrics - Loss: 2.1617, Accuracy: 50.0569
2024-12-30 23:35:39 | INFO | Validation metrics - Loss: 2.1663, Accuracy: 50.2748
2024-12-30 23:35:39 | INFO | Training metrics - Loss: 2.2570, Accuracy: 49.7567, LR: 0.043722
2024-12-30 23:35:39 | INFO | Training metrics - Loss: 2.2486, Accuracy: 49.7878, LR: 0.043722
2024-12-30 23:35:39 | INFO | Training metrics - Loss: 2.2482, Accuracy: 49.7732, LR: 0.043722
2024-12-30 23:35:40 | INFO | 
==================== Epoch 16 ====================
2024-12-30 23:35:40 | INFO | 
==================== Epoch 16 ====================
2024-12-30 23:35:40 | INFO | 
==================== Epoch 16 ====================
2024-12-30 23:51:06 | INFO | New best validation accuracy: 54.4494. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch16-acc54.4494.ckpt
2024-12-30 23:51:06 | INFO | New best validation accuracy: 53.8419. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch16-acc53.8419.ckpt
2024-12-30 23:51:06 | INFO | New best validation accuracy: 53.9694. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch16-acc53.9694.ckpt
2024-12-30 23:51:06 | INFO | Validation metrics - Loss: 1.9346, Accuracy: 54.4494
2024-12-30 23:51:06 | INFO | Validation metrics - Loss: 1.9849, Accuracy: 53.8419
2024-12-30 23:51:06 | INFO | Validation metrics - Loss: 1.9519, Accuracy: 53.9694
2024-12-30 23:51:06 | INFO | Training metrics - Loss: 2.2034, Accuracy: 50.5866, LR: 0.048047
2024-12-30 23:51:06 | INFO | Training metrics - Loss: 2.2021, Accuracy: 50.6620, LR: 0.048047
2024-12-30 23:51:06 | INFO | Training metrics - Loss: 2.1976, Accuracy: 50.7512, LR: 0.048047
2024-12-30 23:51:07 | INFO | 
==================== Epoch 17 ====================
2024-12-30 23:51:07 | INFO | 
==================== Epoch 17 ====================
2024-12-30 23:51:07 | INFO | 
==================== Epoch 17 ====================
2024-12-31 00:06:32 | INFO | Validation metrics - Loss: 2.0879, Accuracy: 51.8579
2024-12-31 00:06:32 | INFO | Training metrics - Loss: 2.1537, Accuracy: 51.6303, LR: 0.052515
2024-12-31 00:06:32 | INFO | Validation metrics - Loss: 2.0703, Accuracy: 51.9888
2024-12-31 00:06:32 | INFO | Training metrics - Loss: 2.1506, Accuracy: 51.6788, LR: 0.052515
2024-12-31 00:06:32 | INFO | Validation metrics - Loss: 2.1262, Accuracy: 51.3433
2024-12-31 00:06:32 | INFO | Training metrics - Loss: 2.1601, Accuracy: 51.4417, LR: 0.052515
2024-12-31 00:06:33 | INFO | 
==================== Epoch 18 ====================
2024-12-31 00:06:33 | INFO | 
==================== Epoch 18 ====================
2024-12-31 00:06:33 | INFO | 
==================== Epoch 18 ====================
2024-12-31 00:22:00 | INFO | Validation metrics - Loss: 1.9694, Accuracy: 53.4628
2024-12-31 00:22:00 | INFO | Training metrics - Loss: 2.1188, Accuracy: 52.2397, LR: 0.057105
2024-12-31 00:22:00 | INFO | Validation metrics - Loss: 1.9600, Accuracy: 53.9269
2024-12-31 00:22:00 | INFO | Training metrics - Loss: 2.1208, Accuracy: 52.2179, LR: 0.057105
2024-12-31 00:22:00 | INFO | Validation metrics - Loss: 2.0230, Accuracy: 53.2201
2024-12-31 00:22:00 | INFO | Training metrics - Loss: 2.1199, Accuracy: 52.1235, LR: 0.057105

```

Spot instance stopped at epoch 18. The training was resumed from epoch 19.

```
2024-12-31 03:44:31 | INFO | Starting training
2024-12-31 03:44:31 | INFO | Resuming training from checkpoint: /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch18-acc53.7369.ckpt
2024-12-31 03:44:38 | INFO | 
==================== Epoch 19 ====================
2024-12-31 03:44:38 | INFO | 
==================== Epoch 19 ====================
2024-12-31 03:44:38 | INFO | 
==================== Epoch 19 ====================
2024-12-31 04:00:01 | INFO | New best validation accuracy: 53.7891. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch19-acc53.7891.ckpt
2024-12-31 04:00:01 | INFO | New best validation accuracy: 53.0839. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch19-acc53.0839.ckpt
2024-12-31 04:00:01 | INFO | New best validation accuracy: 52.8830. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch19-acc52.8830.ckpt
2024-12-31 04:00:01 | INFO | Validation metrics - Loss: 1.9732, Accuracy: 53.7891
2024-12-31 04:00:01 | INFO | Validation metrics - Loss: 2.0226, Accuracy: 53.0839
2024-12-31 04:00:01 | INFO | Validation metrics - Loss: 1.9979, Accuracy: 52.8830
2024-12-31 04:00:01 | INFO | Training metrics - Loss: 2.0741, Accuracy: 53.1357, LR: 0.061798
2024-12-31 04:00:01 | INFO | Training metrics - Loss: 2.0917, Accuracy: 52.7249, LR: 0.061798
2024-12-31 04:00:01 | INFO | Training metrics - Loss: 2.0853, Accuracy: 52.8417, LR: 0.061798
2024-12-31 04:00:01 | INFO | 
==================== Epoch 20 ====================
2024-12-31 04:00:01 | INFO | 
==================== Epoch 20 ====================
2024-12-31 04:00:01 | INFO | 
==================== Epoch 20 ====================
2024-12-31 04:15:18 | INFO | New best validation accuracy: 57.6561. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch20-acc57.6561.ckpt
2024-12-31 04:15:18 | INFO | New best validation accuracy: 57.3452. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch20-acc57.3452.ckpt
2024-12-31 04:15:18 | INFO | New best validation accuracy: 56.6293. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch20-acc56.6293.ckpt
2024-12-31 04:15:18 | INFO | Validation metrics - Loss: 1.7853, Accuracy: 57.6561
2024-12-31 04:15:18 | INFO | Validation metrics - Loss: 1.8357, Accuracy: 56.6293
2024-12-31 04:15:18 | INFO | Validation metrics - Loss: 1.8013, Accuracy: 57.3452
2024-12-31 04:15:18 | INFO | Training metrics - Loss: 2.0624, Accuracy: 53.2954, LR: 0.066574
2024-12-31 04:15:18 | INFO | Training metrics - Loss: 2.0594, Accuracy: 53.3749, LR: 0.066574
2024-12-31 04:15:18 | INFO | Training metrics - Loss: 2.0538, Accuracy: 53.4419, LR: 0.066574
2024-12-31 04:15:19 | INFO | 
==================== Epoch 21 ====================
2024-12-31 04:15:19 | INFO | 
==================== Epoch 21 ====================
2024-12-31 04:15:19 | INFO | 
==================== Epoch 21 ====================
2024-12-31 04:30:44 | INFO | Validation metrics - Loss: 1.9887, Accuracy: 53.1145
2024-12-31 04:30:44 | INFO | Training metrics - Loss: 2.0312, Accuracy: 53.9371, LR: 0.071412
2024-12-31 04:30:44 | INFO | Validation metrics - Loss: 1.9870, Accuracy: 52.3860
2024-12-31 04:30:44 | INFO | Training metrics - Loss: 2.0323, Accuracy: 53.9449, LR: 0.071412
2024-12-31 04:30:44 | INFO | Validation metrics - Loss: 2.0570, Accuracy: 52.1309
2024-12-31 04:30:44 | INFO | Training metrics - Loss: 2.0259, Accuracy: 54.0235, LR: 0.071412
2024-12-31 04:30:44 | INFO | 
==================== Epoch 22 ====================
2024-12-31 04:30:44 | INFO | 
==================== Epoch 22 ====================
2024-12-31 04:30:44 | INFO | 
==================== Epoch 22 ====================
2024-12-31 04:46:09 | INFO | Validation metrics - Loss: 2.0993, Accuracy: 51.5699
2024-12-31 04:46:09 | INFO | Training metrics - Loss: 2.0138, Accuracy: 54.3546, LR: 0.076292
2024-12-31 04:46:09 | INFO | Validation metrics - Loss: 2.0881, Accuracy: 52.1252
2024-12-31 04:46:09 | INFO | Training metrics - Loss: 2.0033, Accuracy: 54.3870, LR: 0.076292
2024-12-31 04:46:10 | INFO | Validation metrics - Loss: 2.1303, Accuracy: 51.4553
2024-12-31 04:46:10 | INFO | Training metrics - Loss: 1.9990, Accuracy: 54.5605, LR: 0.076292
2024-12-31 04:46:10 | INFO | 
==================== Epoch 23 ====================
2024-12-31 04:46:10 | INFO | 
==================== Epoch 23 ====================
2024-12-31 04:46:10 | INFO | 
==================== Epoch 23 ====================
2024-12-31 05:01:36 | INFO | New best validation accuracy: 57.8491. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch23-acc57.8491.ckpt
2024-12-31 05:01:36 | INFO | New best validation accuracy: 58.0629. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch23-acc58.0629.ckpt
2024-12-31 05:01:36 | INFO | New best validation accuracy: 58.3078. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch23-acc58.3078.ckpt
2024-12-31 05:01:36 | INFO | Validation metrics - Loss: 1.7430, Accuracy: 57.8491
2024-12-31 05:01:36 | INFO | Validation metrics - Loss: 1.7767, Accuracy: 58.0629
2024-12-31 05:01:36 | INFO | Validation metrics - Loss: 1.7335, Accuracy: 58.3078
2024-12-31 05:01:36 | INFO | Training metrics - Loss: 1.9882, Accuracy: 54.7568, LR: 0.081192
2024-12-31 05:01:36 | INFO | Training metrics - Loss: 1.9842, Accuracy: 54.9277, LR: 0.081192
2024-12-31 05:01:36 | INFO | Training metrics - Loss: 1.9862, Accuracy: 54.7949, LR: 0.081192
2024-12-31 05:01:37 | INFO | 
==================== Epoch 24 ====================
2024-12-31 05:01:37 | INFO | 
==================== Epoch 24 ====================
2024-12-31 05:01:37 | INFO | 
==================== Epoch 24 ====================
2024-12-31 05:17:02 | INFO | Validation metrics - Loss: 1.9623, Accuracy: 53.8405
2024-12-31 05:17:02 | INFO | Training metrics - Loss: 1.9621, Accuracy: 55.2636, LR: 0.086093
2024-12-31 05:17:02 | INFO | Validation metrics - Loss: 2.0021, Accuracy: 53.7206
2024-12-31 05:17:02 | INFO | Training metrics - Loss: 1.9690, Accuracy: 55.1742, LR: 0.086093
2024-12-31 05:17:03 | INFO | Validation metrics - Loss: 1.9620, Accuracy: 53.9312
2024-12-31 05:17:03 | INFO | Training metrics - Loss: 1.9622, Accuracy: 55.2364, LR: 0.086093
2024-12-31 05:17:03 | INFO | 
==================== Epoch 25 ====================
2024-12-31 05:17:03 | INFO | 
==================== Epoch 25 ====================
2024-12-31 05:17:03 | INFO | 
==================== Epoch 25 ====================
2024-12-31 05:32:29 | INFO | Validation metrics - Loss: 1.8236, Accuracy: 56.1703
2024-12-31 05:32:29 | INFO | Training metrics - Loss: 1.9477, Accuracy: 55.5365, LR: 0.090972
2024-12-31 05:32:29 | INFO | Validation metrics - Loss: 1.8289, Accuracy: 56.5609
2024-12-31 05:32:29 | INFO | Training metrics - Loss: 1.9497, Accuracy: 55.4886, LR: 0.090972
2024-12-31 05:32:30 | INFO | Validation metrics - Loss: 1.8684, Accuracy: 56.0909
2024-12-31 05:32:30 | INFO | Training metrics - Loss: 1.9475, Accuracy: 55.4045, LR: 0.090972
2024-12-31 05:32:30 | INFO | 
==================== Epoch 26 ====================
2024-12-31 05:32:30 | INFO | 
==================== Epoch 26 ====================
2024-12-31 05:32:30 | INFO | 
==================== Epoch 26 ====================
2024-12-31 05:47:56 | INFO | Validation metrics - Loss: 1.8694, Accuracy: 55.5053
2024-12-31 05:47:56 | INFO | Training metrics - Loss: 1.9321, Accuracy: 55.8532, LR: 0.095808
2024-12-31 05:47:56 | INFO | Validation metrics - Loss: 1.9128, Accuracy: 55.2758
2024-12-31 05:47:56 | INFO | Training metrics - Loss: 1.9274, Accuracy: 55.8532, LR: 0.095808
2024-12-31 05:47:56 | INFO | Validation metrics - Loss: 1.8869, Accuracy: 55.2386
2024-12-31 05:47:56 | INFO | Training metrics - Loss: 1.9378, Accuracy: 55.6994, LR: 0.095808
2024-12-31 05:47:56 | INFO | 
==================== Epoch 27 ====================
2024-12-31 05:47:56 | INFO | 
==================== Epoch 27 ====================
2024-12-31 05:47:56 | INFO | 
==================== Epoch 27 ====================
2024-12-31 06:03:20 | INFO | Validation metrics - Loss: 1.9324, Accuracy: 54.3185
2024-12-31 06:03:20 | INFO | Training metrics - Loss: 1.9220, Accuracy: 56.0392, LR: 0.100582
2024-12-31 06:03:20 | INFO | Validation metrics - Loss: 1.9197, Accuracy: 54.9437
2024-12-31 06:03:20 | INFO | Training metrics - Loss: 1.9187, Accuracy: 56.1005, LR: 0.100582
2024-12-31 06:03:21 | INFO | Validation metrics - Loss: 1.9710, Accuracy: 54.4381
2024-12-31 06:03:21 | INFO | Training metrics - Loss: 1.9181, Accuracy: 55.9914, LR: 0.100582
2024-12-31 06:03:21 | INFO | 
==================== Epoch 28 ====================
2024-12-31 06:03:21 | INFO | 
==================== Epoch 28 ====================
2024-12-31 06:03:21 | INFO | 
==================== Epoch 28 ====================
2024-12-31 06:18:45 | INFO | Validation metrics - Loss: 1.7632, Accuracy: 58.0560
2024-12-31 06:18:45 | INFO | Training metrics - Loss: 1.9053, Accuracy: 56.2942, LR: 0.105272
2024-12-31 06:18:45 | INFO | Validation metrics - Loss: 1.7670, Accuracy: 57.6580
2024-12-31 06:18:45 | INFO | Training metrics - Loss: 1.8973, Accuracy: 56.4288, LR: 0.105272
2024-12-31 06:18:45 | INFO | Validation metrics - Loss: 1.8114, Accuracy: 57.3927
2024-12-31 06:18:45 | INFO | Training metrics - Loss: 1.9056, Accuracy: 56.3600, LR: 0.105272
2024-12-31 06:18:45 | INFO | 
==================== Epoch 29 ====================
2024-12-31 06:18:45 | INFO | 
==================== Epoch 29 ====================
2024-12-31 06:18:45 | INFO | 
==================== Epoch 29 ====================
2024-12-31 06:34:10 | INFO | New best validation accuracy: 59.4962. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch29-acc59.4962.ckpt
2024-12-31 06:34:10 | INFO | New best validation accuracy: 59.1155. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch29-acc59.1155.ckpt
2024-12-31 06:34:10 | INFO | Validation metrics - Loss: 1.7072, Accuracy: 59.1155
2024-12-31 06:34:10 | INFO | Validation metrics - Loss: 1.6957, Accuracy: 59.4962
2024-12-31 06:34:10 | INFO | New best validation accuracy: 58.7296. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch29-acc58.7296.ckpt
2024-12-31 06:34:10 | INFO | Validation metrics - Loss: 1.7346, Accuracy: 58.7296
2024-12-31 06:34:10 | INFO | Training metrics - Loss: 1.8962, Accuracy: 56.4786, LR: 0.109859
2024-12-31 06:34:10 | INFO | Training metrics - Loss: 1.8973, Accuracy: 56.4407, LR: 0.109859
2024-12-31 06:34:10 | INFO | Training metrics - Loss: 1.8878, Accuracy: 56.6092, LR: 0.109859
2024-12-31 06:34:10 | INFO | 
==================== Epoch 30 ====================
2024-12-31 06:34:10 | INFO | 
==================== Epoch 30 ====================
2024-12-31 06:34:10 | INFO | 
==================== Epoch 30 ====================
2024-12-31 06:49:34 | INFO | Validation metrics - Loss: 1.7761, Accuracy: 57.6278
2024-12-31 06:49:34 | INFO | Training metrics - Loss: 1.8800, Accuracy: 56.9320, LR: 0.114322
2024-12-31 06:49:34 | INFO | Validation metrics - Loss: 1.8326, Accuracy: 56.9466
2024-12-31 06:49:34 | INFO | Training metrics - Loss: 1.8818, Accuracy: 56.8671, LR: 0.114322
2024-12-31 06:49:34 | INFO | Validation metrics - Loss: 1.7806, Accuracy: 57.2611
2024-12-31 06:49:34 | INFO | Training metrics - Loss: 1.8773, Accuracy: 56.8807, LR: 0.114322
2024-12-31 06:49:35 | INFO | 
==================== Epoch 31 ====================
2024-12-31 06:49:35 | INFO | 
==================== Epoch 31 ====================
2024-12-31 06:49:35 | INFO | 
==================== Epoch 31 ====================
2024-12-31 07:04:58 | INFO | Validation metrics - Loss: 1.7249, Accuracy: 58.7084
2024-12-31 07:04:58 | INFO | Training metrics - Loss: 1.8715, Accuracy: 57.0733, LR: 0.118644
2024-12-31 07:04:58 | INFO | Validation metrics - Loss: 1.7332, Accuracy: 58.5808
2024-12-31 07:04:58 | INFO | Training metrics - Loss: 1.8632, Accuracy: 57.0447, LR: 0.118644
2024-12-31 07:04:59 | INFO | Validation metrics - Loss: 1.7720, Accuracy: 58.0467
2024-12-31 07:04:59 | INFO | Training metrics - Loss: 1.8673, Accuracy: 57.0540, LR: 0.118644
2024-12-31 07:04:59 | INFO | 
==================== Epoch 32 ====================
2024-12-31 07:04:59 | INFO | 
==================== Epoch 32 ====================
2024-12-31 07:04:59 | INFO | 
==================== Epoch 32 ====================
2024-12-31 07:20:24 | INFO | Validation metrics - Loss: 1.9685, Accuracy: 54.1362
2024-12-31 07:20:24 | INFO | Training metrics - Loss: 1.8604, Accuracy: 57.2186, LR: 0.122804
2024-12-31 07:20:24 | INFO | Validation metrics - Loss: 1.9818, Accuracy: 53.6071
2024-12-31 07:20:24 | INFO | Training metrics - Loss: 1.8652, Accuracy: 57.0896, LR: 0.122804
2024-12-31 07:20:24 | INFO | Validation metrics - Loss: 2.0293, Accuracy: 53.2002
2024-12-31 07:20:24 | INFO | Training metrics - Loss: 1.8615, Accuracy: 57.1877, LR: 0.122804
2024-12-31 07:20:24 | INFO | 
==================== Epoch 33 ====================
2024-12-31 07:20:24 | INFO | 
==================== Epoch 33 ====================
2024-12-31 07:20:24 | INFO | 
==================== Epoch 33 ====================
2024-12-31 07:35:47 | INFO | Validation metrics - Loss: 1.7289, Accuracy: 58.9811
2024-12-31 07:35:47 | INFO | Training metrics - Loss: 1.8523, Accuracy: 57.3063, LR: 0.126785
2024-12-31 07:35:48 | INFO | Validation metrics - Loss: 1.7111, Accuracy: 58.9951
2024-12-31 07:35:48 | INFO | Training metrics - Loss: 1.8489, Accuracy: 57.4578, LR: 0.126785
```

Spot instance stopped at epoch 33. The training was resumed from epoch 34.

```
2024-12-31 09:14:20 | INFO | Resuming training from checkpoint: /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch33-acc59.0611.ckpt
2024-12-31 09:14:27 | INFO | 
==================== Epoch 34 ====================
2024-12-31 09:14:27 | INFO | 
==================== Epoch 34 ====================
2024-12-31 09:14:27 | INFO | 
==================== Epoch 34 ====================
2024-12-31 09:30:02 | INFO | New best validation accuracy: 57.5478. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch34-acc57.5478.ckpt
2024-12-31 09:30:02 | INFO | New best validation accuracy: 57.4780. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch34-acc57.4780.ckpt
2024-12-31 09:30:02 | INFO | New best validation accuracy: 57.5305. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch34-acc57.5305.ckpt
2024-12-31 09:30:02 | INFO | Validation metrics - Loss: 1.8058, Accuracy: 57.5478
2024-12-31 09:30:02 | INFO | Validation metrics - Loss: 1.7764, Accuracy: 57.4780
2024-12-31 09:30:02 | INFO | Validation metrics - Loss: 1.7724, Accuracy: 57.5305
2024-12-31 09:30:02 | INFO | Training metrics - Loss: 1.8466, Accuracy: 57.5003, LR: 0.130571
2024-12-31 09:30:02 | INFO | Training metrics - Loss: 1.8412, Accuracy: 57.6840, LR: 0.130571
2024-12-31 09:30:02 | INFO | Training metrics - Loss: 1.8378, Accuracy: 57.6496, LR: 0.130571
2024-12-31 09:30:02 | INFO | 
==================== Epoch 35 ====================
2024-12-31 09:30:02 | INFO | 
==================== Epoch 35 ====================
2024-12-31 09:30:02 | INFO | 
==================== Epoch 35 ====================
2024-12-31 09:45:27 | INFO | New best validation accuracy: 58.3925. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch35-acc58.3925.ckpt
2024-12-31 09:45:27 | INFO | New best validation accuracy: 58.7356. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch35-acc58.7356.ckpt
2024-12-31 09:45:27 | INFO | New best validation accuracy: 58.5852. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch35-acc58.5852.ckpt
2024-12-31 09:45:27 | INFO | Validation metrics - Loss: 1.7088, Accuracy: 58.5852
2024-12-31 09:45:27 | INFO | Validation metrics - Loss: 1.7461, Accuracy: 58.3925
2024-12-31 09:45:27 | INFO | Validation metrics - Loss: 1.7092, Accuracy: 58.7356
2024-12-31 09:45:27 | INFO | Training metrics - Loss: 1.8264, Accuracy: 57.8975, LR: 0.134145
2024-12-31 09:45:27 | INFO | Training metrics - Loss: 1.8328, Accuracy: 57.7386, LR: 0.134145
2024-12-31 09:45:27 | INFO | Training metrics - Loss: 1.8364, Accuracy: 57.6758, LR: 0.134145
2024-12-31 09:45:28 | INFO | 
==================== Epoch 36 ====================
2024-12-31 09:45:28 | INFO | 
==================== Epoch 36 ====================
2024-12-31 09:45:28 | INFO | 
==================== Epoch 36 ====================
2024-12-31 11:27:23 | INFO | New best validation accuracy: 59.8464. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch36-acc59.8464.ckpt
2024-12-31 11:27:23 | INFO | New best validation accuracy: 60.0911. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch36-acc60.0911.ckpt
2024-12-31 11:27:23 | INFO | New best validation accuracy: 60.8130. Saved checkpoint to /home/ec2-user/ebs/volumes/era_session9/resnet50-epoch36-acc60.8130.ckpt
2024-12-31 11:27:23 | INFO | Validation metrics - Loss: 1.6302, Accuracy: 60.8130
2024-12-31 11:27:23 | INFO | Validation metrics - Loss: 1.6763, Accuracy: 59.8464
2024-12-31 11:27:23 | INFO | Validation metrics - Loss: 1.6392, Accuracy: 60.0911
2024-12-31 11:27:23 | INFO | Training metrics - Loss: 1.8200, Accuracy: 58.0436, LR: 0.137491
2024-12-31 11:27:23 | INFO | Training metrics - Loss: 1.8220, Accuracy: 57.9928, LR: 0.137491
2024-12-31 11:27:23 | INFO | Training metrics - Loss: 1.8190, Accuracy: 57.9645, LR: 0.137491
2024-12-31 11:27:23 | INFO | 
```
