
#---------------------------------------------------------------------
# Convert sample tranning classifier from Pytorch documentation page
#into pytorch_lightning model. You can find original  pytorch model here:
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# 
# Find more about pytorch_lightining framework here:
# https://pytorch-lightning.readthedocs.io/en/latest/new-project.html
#**********************************************************************

import logging as log
import argparse
import os
import random
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_lightning as ptl


class CNNModel(ptl.LightningModule):

    def __init__(self, hparams):
        super(CNNModel, self).__init__()
        self.params = hparams
        #in_channels, out_channels, kernel_size
        self.conv1 = nn.Conv2d(self.params.l1conv_in_channels,
                               self.params.l1conv_out_channels, 
                               self.params.l1conv_kernel)

        self.conv2 = nn.Conv2d(self.params.l1conv_out_channels,
                               self.params.l2conv_out_channels,
                               self.params.l2conv_kernel)

        self.pool = nn.MaxPool2d(self.params.maxpool)
        self.fc1 = nn.Linear(self.params.l1Lin_in_features,
                             self.params.l1lin_out_features)
        self.fc2 = nn.Linear(self.params.l1lin_out_features,
                             self.params.l2lin_out_features)
        self.fc3 = nn.Linear(self.params.l2conv_out_channels,
                             self.params.l3lin_out_features)
        
        
        
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.params.l1Lin_in_features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.cross_entropy(y_hat, y)(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),
                              lr = self.params.lr,
                              momentum=self.params.momentum)
        criterion = nn.CrossEntropyLoss()
        return [optimizer], [criterion]

    def __dataloader(self, train):

        transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10(root = self.params.data_root, train = train,
                                   transform=transform, download=True)

        #when using multinode (multiple computers) (ddp), we need to add data sampler
        #note each node can have 1 or more GPUs 
        train_sampler = None
        batch_size = self.params.batch_size

        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)
        should_shuffle = train_sampler is None
        loader = DataLoader(dataset= dataset,
                            batch_size=batch_size,
                            shuffle=should_shuffle,
                            sampler=train_sampler,
                            num_workers=self.params.num_workers)
        
        return loader
        


    @ptl.data_loader
    def tng_dataloader(self):
        log.info('Training data loader called')
        return self.__dataloader(train=True)

    @ptl.data_loader
    def val_dataloader(self):
        log.info('Validation data loader called')
        return self.__dataloader(train=False)


    @ptl.data_loader
    def test_dataloader(self):
        log.info('Test data loader called')
        return self.__dataloader(train=False)
       
    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = argparse.ArgumentParser(parents=[parent_parser])

        #network partms
        parser.add_argument('--l1conv_in_channels', default=3, help='conv layer l1 input channels', type=int)
        parser.add_argument('--l1conv_out_channels', default=6, help='conv layer 1 output channels', type=int)
        parser.add_argument('--l1conv_kernel', default=5, help='l1 kernel size', type=int)

        parser.add_argument('--l2conv_out_channels', default=16, help='layer 2 output channels', type= int)
        parser.add_argument('--l2conv_kernel', default=5, help='layer 2 kernel size', type = int)
        parser.add_argument('--maxpool', default=2, help='maxpool2d kernel size', type=int)
        parser.add_argument('--l1Lin_in_features', default=16*5*5, help='infeatures of First linear layer', type=int)
        parser.add_argument('--l1lin_out_features', default=120, help='outfeatures of first linear layer', type=int)
        parser.add_argument('--l2lin_out_features', default=84, help='outfeatures of send linear layer', type=int)
        parser.add_argument('--l3lin_out_features', default=10, help='total number of output classes.', type=int)
        parser.add_argument('--lr', default=0.001, help='learning rate')
        parser.add_argument('--momentum', default=0.9, help='momentum')
        parser.add_argument('--epochs', default=2, help='epochs', type=int)

        #data 
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'CIFAR10'), type= str, 
                            help='data path')
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--num_workers', default=2, type=int)
        return parser


def get_args():
    root_dir= os.path.dirname(os.path.realpath(__file__))
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--gpus', type=int, default=1, help='how many gpu')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use-16bit', dest='use_16bit', action='store_true',
                               help='if true uses 16 bit precision')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')

    parser = CNNModel.add_model_specific_args(parent_parser, root_dir)
    hparams = parser.parse_args(args=[])
    return hparams
    log.info('ended get args method')


def main(hparams):

    
    # 1 Initialize lightining model
    model = CNNModel(hparams)
    # 2 initialize trainer
    trainer = ptl.Trainer(
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        distributed_backend=hparams.distributed_backend,
        use_amp=hparams.use_16bit
    )
    # 3 train the model
    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)
    
    

if __name__ == '__main__':

    main(get_args())
    log.info('run main')