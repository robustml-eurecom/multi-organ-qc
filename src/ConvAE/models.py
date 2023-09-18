import os
import numpy as np
from abc import abstractmethod, ABC
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import softplus
import pytorch_lightning as pl

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
        

class VAEAnomalyDetection(pl.LightningModule, ABC):
    """
    Variational Autoencoder (VAE) for anomaly detection. The model learns a low-dimensional representation of the input
    data using an encoder-decoder architecture, and uses the learned representation to detect anomalies.

    The model is trained to minimize the Kullback-Leibler (KL) divergence between the learned distribution of the latent
    variables and the prior distribution (a standard normal distribution). It is also trained to maximize the likelihood
    of the input data under the learned distribution.

    This implementation uses PyTorch Lightning to simplify training and improve reproducibility.
    """

    def __init__(self, input_size: int, latent_size: int, L: int = 10, lr: float = 1e-3, log_steps: int = 1_000):
        """
        Initializes the VAEAnomalyDetection model.

        Args:
            input_size (int): Number of input features.
            latent_size (int): Size of the latent space.
            L (int, optional): Number of samples in the latent space to detect the anomaly. Defaults to 10.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            log_steps (int, optional): Number of steps between each logging. Defaults to 1_000.
        """
        super().__init__()
        self.L = L
        self.lr = lr
        self.input_size = input_size
        self.latent_size = latent_size
        self.encoder = self.make_encoder(input_size, latent_size)
        self.decoder = self.make_decoder(latent_size, input_size)
        self.prior = Normal(0, 1)
        self.log_steps = log_steps

    @abstractmethod
    def make_encoder(self, input_size: int, latent_size: int) -> nn.Module:
        """
        Abstract method to create the encoder network.

        Args:
            input_size (int): Number of input features.
            latent_size (int): Size of the latent space.

        Returns:
            nn.Module: Encoder network.
        """
        pass

    @abstractmethod
    def make_decoder(self, latent_size: int, output_size: int) -> nn.Module:
        """
        Abstract method to create the decoder network.

        Args:
            latent_size (int): Size of the latent space.
            output_size (int): Number of output features.

        Returns:
            nn.Module: Decoder network.
        """
        pass

    def forward(self, x: torch.Tensor) -> dict:
        """
        Computes the forward pass of the model and returns the loss and other relevant information.

        Args:
            x (torch.Tensor): Input data. Shape [batch_size, num_features].

        Returns:
            Dictionary containing:
            - loss: Total loss.
            - kl: KL-divergence loss.
            - recon_loss: Reconstruction loss.
            - recon_mu: Mean of the reconstructed input.
            - recon_sigma: Standard deviation of the reconstructed input.
            - latent_dist: Distribution of the latent space.
            - latent_mu: Mean of the latent space.
            - latent_sigma: Standard deviation of the latent space.
            - z: Sampled latent space.

        """
        pred_result = self.predict(x)
        x = x.unsqueeze(0)  # unsqueeze to broadcast input across sample dimension (L)
        log_lik = Normal(pred_result['recon_mu'], pred_result['recon_sigma']).log_prob(x).mean(
            dim=0)  # average over sample dimension
        log_lik = log_lik.mean(dim=0).sum()
        kl = kl_divergence(pred_result['latent_dist'], self.prior).mean(dim=0).sum()
        loss = kl - log_lik
        return dict(loss=loss, kl=kl, recon_loss=log_lik, **pred_result)

    def predict(self, x) -> dict:
        """
        Compute the output of the VAE. Does not compute the loss compared to the forward method.

        Args:
            x: Input tensor of shape [batch_size, input_size].

        Returns:
            Dictionary containing:
            - latent_dist: Distribution of the latent space.
            - latent_mu: Mean of the latent space.
            - latent_sigma: Standard deviation of the latent space.
            - recon_mu: Mean of the reconstructed input.
            - recon_sigma: Standard deviation of the reconstructed input.
            - z: Sampled latent space.

        """
        batch_size = len(x)
        latent_mu, latent_sigma = self.encoder(x).chunk(2, dim=1) #both with size [batch_size, latent_size]
        latent_sigma = softplus(latent_sigma)
        dist = Normal(latent_mu, latent_sigma)
        z = dist.rsample([self.L])  # shape: [L, batch_size, latent_size]
        z = z.view(self.L * batch_size, self.latent_size)
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        recon_mu = recon_mu.view(self.L, *x.shape)
        recon_sigma = recon_sigma.view(self.L, *x.shape)
        return dict(latent_dist=dist, latent_mu=latent_mu,
                    latent_sigma=latent_sigma, recon_mu=recon_mu,
                    recon_sigma=recon_sigma, z=z)

    def is_anomaly(self, x: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        """
        Determines if input samples are anomalous based on a given threshold.
        
        Args:
            x: Input tensor of shape (batch_size, num_features).
            alpha: Anomaly threshold. Values with probability lower than alpha are considered anomalous.
        
        Returns:
            A binary tensor of shape (batch_size,) where `True` represents an anomalous sample and `False` represents a 
            normal sample.
        """
        p = self.reconstructed_probability(x)
        return p < alpha

    def reconstructed_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the probability density of the input samples under the learned
        distribution of reconstructed data.

        Args:
            x: Input data tensor of shape (batch_size, num_features).

        Returns:
            A tensor of shape (batch_size,) containing the probability densities of
            the input samples under the learned distribution of reconstructed data.
        """
        with torch.no_grad():
            pred = self.predict(x)
        recon_dist = Normal(pred['recon_mu'], pred['recon_sigma'])
        x = x.unsqueeze(0)
        p = recon_dist.log_prob(x).exp().mean(dim=0).mean(dim=-1)  # vector of shape [batch_size]
        return p

    def generate(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generates a batch of samples from the learned prior distribution.

        Args:
            batch_size: Number of samples to generate.

        Returns:
            A tensor of shape (batch_size, num_features) containing the generated
            samples.
        """
        z = self.prior.sample((batch_size, self.latent_size))
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        return recon_mu + recon_sigma * torch.rand_like(recon_sigma)
    
    
    def training_step(self, batch, batch_idx):
        x = batch
        loss = self.forward(x)
        if self.global_step % self.log_steps == 0:
            self.log('train/loss', loss['loss'])
            self.log('train/loss_kl', loss['kl'], prog_bar=False)
            self.log('train/loss_recon', loss['recon_loss'], prog_bar=False)
            self._log_norm()

        return loss
    

    def validation_step(self, batch, batch_idx):
        x = batch
        loss = self.forward(x)
        self.log('val/loss_epoch', loss['loss'], on_epoch=True)
        self.log('val_kl', loss['kl'], self.global_step)
        self.log('val_recon_loss', loss['recon_loss'], self.global_step)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


    def _log_norm(self):
        norm1 = sum(p.norm(1) for p in self.parameters())
        norm1_grad = sum(p.grad.norm(1) for p in self.parameters() if p.grad is not None)
        self.log('norm1_params', norm1)
        self.log('norm1_grad', norm1_grad)

class VAEAnomalyTabular(VAEAnomalyDetection):

    def make_encoder(self, input_size, latent_size):
        """
        Simple encoder for tabular data.
        If you want to feed image to a VAE make another encoder function with Conv2d instead of Linear layers.
        :param input_size: number of input variables
        :param latent_size: number of output variables i.e. the size of the latent space since it's the encoder of a VAE
        :return: The untrained encoder model
        """
        return nn.Sequential(
            nn.Linear(input_size, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, latent_size * 2)
            # times 2 because this is the concatenated vector of latent mean and variance
        )

    def make_decoder(self, latent_size, output_size):
        """
        Simple decoder for tabular data.
        :param latent_size: size of input latent space
        :param output_size: number of output parameters. Must have the same value of input_size
        :return: the untrained decoder
        """
        return nn.Sequential(
            nn.Linear(latent_size, 200),
            nn.ReLU(),
            nn.Linear(200, 500),
            nn.ReLU(),
            nn.Linear(500, output_size * 2)  # times 2 because this is the concatenated vector of reconstructed mean and variance
        )

        
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