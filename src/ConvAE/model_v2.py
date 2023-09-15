import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from .building_blocks import ConvolutionalBlock
from .loss import Loss
from .metrics import Metrics

import warnings

#use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvAutoencoder(nn.Module):    
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
        self.optimizer = kwargs["optimizer"](
            self.parameters(),
            lr=kwargs["lr"],
            **{k:v for k,v in kwargs.items() if k in ["weight_decay", "momentum"]}
            )

    def init_model(self, kwargs):
        channel_config = [32, 32, 32, 32, 64, 64, 128, 64, 32]
        pool_config = [True, True, True, False, True, False, True, False, False]
        kernel_config = [4 if el else 3 for el in pool_config]
        stride_config = [2 if el else 1 for el in pool_config]
        
        # Create first conv encoder layers using in_channels
        self.first_conv = ConvolutionalBlock(kwargs["in_channels"], channel_config[0], 
                                             activation=kwargs['activation'], pooling=False,
                                             kernel_size=kernel_config[0], stride=stride_config[0])
        
        # Create encoder layers using channel_config
        self.encoders = nn.ModuleList()
        self.encoders.append(self.first_conv)
        for i in range(0, len(channel_config) - 1):
            encoder_block = ConvolutionalBlock(channel_config[i], channel_config[i + 1], pooling=False,
                                                activation=kwargs['activation'], kernel_size=kernel_config[i + 1],
                                                stride=stride_config[i + 1])
            self.encoders.append(encoder_block)

        # Create shared convolutional layers for the latent space. 
        # We are preserving the spatial dimensions of the input.
        self.latent_conv = nn.Sequential(
            ConvolutionalBlock(channel_config[-1], kwargs["latent_channels"], 
                               activation=kwargs['activation'], kernel_size=4, stride=2,
                               pooling=False, is_dropout=False),
            ConvolutionalBlock(kwargs["latent_channels"], channel_config[-1], 
                               kernel_size=4, transpose=True, activation=kwargs['activation'], 
                               stride=2, pooling=False)
        )

        channel_config.reverse(), kernel_config.reverse(), stride_config.reverse()
        
        # Create decoder layers using channel_config in reverse order
        self.decoders = nn.ModuleList()
        for i in range(0, len(channel_config) - 1):
            decoder_block = ConvolutionalBlock(channel_config[i], channel_config[i + 1], 
                                               kernel_size=kernel_config[i], transpose=True, 
                                               activation=kwargs['activation'], stride=stride_config[i], 
                                               pooling=False)
            self.decoders.append(decoder_block)
        
        # Final convolutional layer to reconstruct the original input and it uses softmax as activation function
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=kwargs["out_channels"], kernel_size=4, stride=2, padding=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Encoder
        encoder_outputs = []
        for encoder_block in self.encoders:
            x = encoder_block(x)
            encoder_outputs.append(x)
            
        # Latent space
        x = self.latent_conv(x)
        
        # Decoder
        for decoder_block in self.decoders:
            x = decoder_block(x)
        
        # Final reconstruction
        reconstruction = self.final_conv(x)

        return reconstruction, encoder_outputs

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)
            
    def training_routine(self, epochs, train_loader, val_loader, ckpt_folder=None):
        if ckpt_folder is not None and not os.path.isdir(ckpt_folder):
            os.mkdir(ckpt_folder)
        history = []
        best_acc = None
        for epoch in tqdm(epochs, desc= 'Epochs progress: '):
            self.train()
            for patient in train_loader:
                for batch in patient:
                    batch = batch.to(device)
                    self.optimizer.zero_grad()
                    reconstruction, _ = self.forward(batch)
                    loss = self.loss_function(reconstruction, batch, epoch)
                    loss.backward()
                    self.optimizer.step()
                    
            self.eval()
            with torch.no_grad():
                result = self.evaluation_routine(val_loader, epoch)
            
            if ckpt_folder is not None and (best_acc is None or result['Total'] < best_acc or epoch%10 == 0):
                ckpt = os.path.join(ckpt_folder,"{:03d}.pth".format(epoch))
                if best_acc is None or result['Total'] < best_acc:
                    best_acc = result['Total']
                    ckpt = ckpt.split(".pth")[0] + "_best.pth"
                torch.save({"AE": self.state_dict(), "AE_optim": self.optimizer.state_dict(), "epoch": epoch}, ckpt)
                clean_old_checkpoints(ckpt_folder)

            self.epoch_end(epoch, result)
            history.append(result)
        return history

    def evaluation_routine(self, val_loader, epoch):
        epoch_summary = {}
        for patient in val_loader:
            gt, reconstruction = [], []
            for batch in patient:
                batch = {"gt": batch.to(device)}
                batch["reconstruction"], _ = self.forward(batch["gt"])
                gt = torch.cat([gt, batch["gt"]], dim=0) if len(gt) > 0 else batch["gt"]
                reconstruction = torch.cat([reconstruction, batch["reconstruction"]], dim=0) if len(reconstruction) > 0 else batch["reconstruction"]
                for k,v in self.loss_function(batch["reconstruction"], batch["gt"], epoch, validation=True).items():
                    if k not in epoch_summary.keys():
                        epoch_summary[k]=[]
                    epoch_summary[k].append(v)
            
            gt = np.argmax(gt.cpu().numpy(), axis=1)
            gt = {"ED": gt[:len(gt)//2], "ES":gt[len(gt)//2:]}
            reconstruction = np.argmax(reconstruction.cpu().numpy(), axis=1)
            reconstruction = {"ED": reconstruction[:len(reconstruction)//2], "ES": reconstruction[len(reconstruction)//2:]}
            for phase in ["ED", "ES"]:
                for k,v in self.metrics(reconstruction[phase], gt[phase]).items():
                    if k not in epoch_summary.keys():
                        epoch_summary[k] = []
                    epoch_summary[k].append(v)
        epoch_summary = {k: np.mean(v) for k,v in epoch_summary.items()}
        return epoch_summary

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
            ckpt = os.path.join(data_path,"checkpoints/", sorted([file for file in os.listdir(os.path.join(data_path,"checkpoints")) if "_best" in file])[-1])
        else:
            ckpt = checkpoint_path
        print("Chosen checkpoint is {} .".format(os.path.split(ckpt)[1]))
        print("###################################")
        ckpt = torch.load(ckpt)

        self.load_state_dict(ckpt["AE"])
        self.optimizer.load_state_dict(ckpt["AE_optim"])
        if eval: self.eval()
        
###############
#Checkpointing#
###############

def clean_old_checkpoints(ckpt_folder:str, best_keep:int=2, total_keep:int=10):
    """
        Is called after each training routine: will keep the {best_keep} best checkpoints, and the other most recent checkpoints for a total of {total_keep checkpoints}. 
        Others will be deleted.
        Example: keep 070_best, 077_best, and 0_78, 0_78, ... and some others

        Parameters
        ----------
            ckpt_folder: str
                The string path to the checkpoints folder
            best_keep: int (default = 2)
                The number of last best checkoints to keep
            total_keep: int (default = 10)
                The total number of checkoints to keep, best and not best included
    """
    assert(best_keep <= total_keep)
    assert(os.path.isdir(ckpt_folder))
    poor_keep = total_keep - best_keep
    poor_ckpts = sorted([file for file in os.listdir(ckpt_folder) if "_best" not in file])
    best_ckpts = sorted([file for file in os.listdir(ckpt_folder) if "_best" in file])
    if len(poor_ckpts)+len(best_ckpts)>total_keep :
        if(len(best_ckpts)>best_keep):
            for file in best_ckpts[:-best_keep] :
                file_path = os.path.join(ckpt_folder, file)
                os.remove(file_path)
        if(len(poor_ckpts)>poor_keep):
            for file in poor_ckpts[:-poor_keep]:
                file_path = os.path.join(ckpt_folder, file)
                os.remove(file_path)