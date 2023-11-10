import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.utils as vutils

from .building_blocks import ConvolutionalBlock
from .cae import clean_old_checkpoints
from models.loss import Loss
from models.metrics import Metrics

import warnings

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmallConvAutoencoder(nn.Module):    
    def __init__(self, keys, **kwargs):
        """
        A convolutional autoencoder for image data.

        This autoencoder consists of encoder and decoder branches with shared convolutional layers in the latent space.

        Args:
            in_channels (int): Number of input channels.
            channel_config (list): List of integers specifying the number of output channels for each convolutional block.
            stride_config (list): List of integers specifying the stride for each convolutional block.
            latent_channels (int): Dimensionality in the latent space.
            optimizer parameters:   
                - optimizer (torch.optim): Optimizer to use for training.
                - lr (float): Learning rate.
                - weight_decay (float): Weight decay.

        Attributes:
            encoders (nn.ModuleList): List of encoder modules.
            latent_conv (nn.Sequential): Shared convolutional layers in the latent space.
            decoders (nn.ModuleList): List of decoder modules.

        Example:
            >>> in_channels = 3
            >>> channel_config = [64, 128, 256, 128, 64]
            >>> latent_channels = 32
            >>> autoencoder = ConvAutoencoder(in_channels, channel_config, latent_channels)
            >>> input_tensor = torch.randn(1, 3, 128, 128)
            >>> output, enc_outputs = autoencoder(input_tensor)
        """

        assert all([type(cls) == str for cls in keys]), 'keys parameter must be an array of Strings'
        assert len(keys) >= 2, 'At least two keys are necessary.'
        
        if not all([len(cls)<=3 for cls in keys]): warnings.warn("Using key labels with length >3 will lead to display incoherences")

        super().__init__()

        self.init_model(kwargs)
        self.apply(self.weight_init)
        self.keys = keys
        self.loss_function = Loss(
            kwargs["functions"], 
            kwargs["settling_epochs_BKGDLoss"], 
            kwargs["settling_epochs_BKMSELoss"]
            )
        self.metrics = Metrics(self.keys)
        self.lr = kwargs["lr"]
        self.optimizer = kwargs["optimizer"](
            self.parameters(),
            lr=self.lr,
            **{k:v for k,v in kwargs.items() if k in ["weight_decay", "momentum"]}
            )

    def init_model(self, kwargs):
        channel_config = [16, 32, 64]
        pool_config = [True, True, False]
        kernel_config = [4 if el else 3 for el in pool_config]
        stride_config = [2 if el else 1 for el in pool_config]

        self.first_conv = ConvolutionalBlock(kwargs["in_channels"], channel_config[0], 
                                            activation=kwargs['activation'], kernel_size=kernel_config[0], stride=stride_config[0])

        self.encoders = nn.ModuleList()
        self.encoders.append(self.first_conv)
        for i in range(0, len(channel_config) - 1):
            encoder_block = ConvolutionalBlock(channel_config[i], channel_config[i + 1], 
                                            activation=kwargs['activation'], kernel_size=kernel_config[i + 1],
                                            stride=stride_config[i + 1])
            self.encoders.append(encoder_block)

        self.latent_conv = nn.Sequential(
            ConvolutionalBlock(channel_config[-1], kwargs["latent_channels"], 
                            activation=kwargs['activation'], kernel_size=4, stride=2,
                            is_dropout=False),
            ConvolutionalBlock(kwargs["latent_channels"], channel_config[-1], 
                            kernel_size=4, transpose=True, activation=kwargs['activation'], 
                            stride=2, pooling=False)
        )

        channel_config.reverse(), kernel_config.reverse(), stride_config.reverse()

        self.decoders = nn.ModuleList()
        for i in range(0, len(channel_config) - 1):
            decoder_block = ConvolutionalBlock(channel_config[i], channel_config[i + 1], 
                                            kernel_size=kernel_config[i], transpose=True, 
                                            activation=kwargs['activation'], stride=stride_config[i], 
                                            pooling=False)
            self.decoders.append(decoder_block)

        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=kwargs["out_channels"], kernel_size=4, stride=2, padding=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        encoder_outputs = []
        for encoder_block in self.encoders:
            x = encoder_block(x)
            encoder_outputs.append(x)
        x = self.latent_conv(x)
        for decoder_block in self.decoders:
            x = decoder_block(x)
        reconstruction = self.final_conv(x)
        return reconstruction, encoder_outputs

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)
            
    def training_routine(self, epochs, train_loader, val_loader, ckpt_folder=None):
        if ckpt_folder is not None and not os.path.isdir(ckpt_folder):
            os.makedirs(ckpt_folder)
        history = []
        img_list = []
        best_acc = None
        lr_warmup = 1e-5
        for epoch in tqdm(epochs, desc= 'Epochs progress: '):
            self.train()
            for batch in train_loader:
                #for batch in patient:
                    batch = batch.to(device)
                    self.optimizer.zero_grad()
                    reconstruction, _ = self.forward(batch)
                    loss = self.loss_function(reconstruction.to(device), batch, epoch)
                    loss.backward()
                    self.optimizer.step()
                    
            self.eval()
            with torch.no_grad():
                result, pred = self.evaluation_routine(val_loader, epoch)
                img_list.append(vutils.make_grid(pred.unsqueeze(0)))
            if ckpt_folder is not None and (best_acc is None or result['Total'] < best_acc or epoch%10 == 0):
                ckpt = os.path.join(ckpt_folder,"{:03d}.pth".format(epoch))
                if best_acc is None or result['Total'] < best_acc:
                    best_acc = result['Total']
                    ckpt = ckpt.split(".pth")[0] + "_best.pth"
                torch.save({"smallAE": self.state_dict(), "smallAE_optim": self.optimizer.state_dict(), "epoch": epoch}, ckpt)
                clean_old_checkpoints(ckpt_folder)
            
            
            self.epoch_end(epoch, result)
            history.append(result)
        return history, img_list

    def evaluation_routine(self, val_loader, epoch):
        epoch_summary = {}
        for batch in val_loader:
                gt, reconstruction = [], []
            #for batch in patient:
                batch = {"gt": batch.to(device)}
                batch["reconstruction"], _ = self.forward(batch["gt"])
                gt = torch.cat([gt, batch["gt"]], dim=0) if len(gt) > 0 else batch["gt"]
                reconstruction = torch.cat([reconstruction, batch["reconstruction"]], dim=0) if len(reconstruction) > 0 else batch["reconstruction"]
                
                for k,v in self.loss_function(batch["reconstruction"], batch["gt"], epoch, validation=True).items():
                    if k not in epoch_summary.keys():
                        epoch_summary[k]=[]
                    epoch_summary[k].append(v)
            
                gt = np.argmax(gt.cpu().numpy(), axis=1) if gt.shape[1] > 1 else gt.cpu().numpy()
                reconstruction = np.argmax(reconstruction.cpu().numpy(), axis=1) if reconstruction.shape[1] > 1 else reconstruction.cpu().numpy()
                
                for k,v in self.metrics(reconstruction, gt).items():
                    if k not in epoch_summary.keys():
                        epoch_summary[k] = []
                    epoch_summary[k].append(v)
                    
        epoch_summary = {k: np.mean(v) for k,v in epoch_summary.items()}
        
        return epoch_summary, batch["reconstruction"].argmax(1).int()

    def epoch_end(self,epoch,result):
        print("\033[1mEpoch [{}]\033[0m".format(epoch))
        header, row = "", ""
        for k,v in result.items():
          header += "{:.6}\t".format(k)
          row += "{:.6}\t".format("{:.4f}".format(v))
        print(header)
        print(row)

    def load_checkpoint(self, data_path:os.PathLike='default', checkpoint_path:os.PathLike='default', eval:bool=False ):
        """
        Loads the model checkpoint, by default the last best saved.
        Setting a value for checkpoint_path will override data_path default behaviour.

        Parameters
        ----------
            data_path: os.PathLike (default = 'default')
                The root dir of the data (eg. data/brain), only useful is checkpoint_path isn't provided
            checkpoint_path: os.PathLike (default = 'default')
                If 'default' is set, then last best saved checkpoint will be loaded. Otherwise, provide path to custom checkpoint.
            eval:bool (default = False)
                If set to True, ae.eval() will also be called. Leave to False for training.
        """

        assert data_path != 'default' or checkpoint_path != 'default', "Either data_path or checkpoint_path must be provided"

        if checkpoint_path=='default':
            ckpt = os.path.join(data_path,"checkpoints/small_cae/", sorted([file for file in os.listdir(os.path.join(data_path,"checkpoints/small_cae")) if "_best" in file])[-1])
        else:
            ckpt = checkpoint_path
        print("Chosen checkpoint is {} .".format(os.path.split(ckpt)[1]))
        print("###################################")
        ckpt = torch.load(ckpt)

        self.load_state_dict(ckpt["smallAE"])
        self.optimizer.load_state_dict(ckpt["smallAE_optim"])
        if eval: self.eval()