#%matplotlib inline
import argparse
import os
import random
import yaml
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

from models.loss import Loss
from models.metrics import Metrics
from models.ConvAE.cae import clean_old_checkpoints

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, ngf, c):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.latent_space, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(.2, True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, ngf, 4, 2, 1, bias=False),
            nn.ConvTranspose2d( ngf, ngf, 4, 2, 1, bias=False),
            nn.ConvTranspose2d( ngf, c, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf, c):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 256 x 256``
            nn.Conv2d(c, ndf, 4, 2, 1, bias=False),
            # state size. ``(ndf=64) x 128 x 128``
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            # state size. ``(ndf=64) x 64 x 64``
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            # state size. ``(ndf=64) x 32 x 32``
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8=512) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class DCGAN(nn.Module):
    """
    Deep Convolutional Generative Adversarial Network (DCGAN) implementation.

    Args:
        ngf (int): Number of generator features.
        ndf (int): Number of discriminator features.
        in_channels (int): Number of input channels.
        lr (float): Learning rate.
        latent_channels (int): Dimension of the latent space.
        optimizer (list): List of optimizers for the generator and discriminator.
        functions (dict): Dictionary of loss functions.

    Attributes:
        netG (nn.Module): Generator network.
        netD (nn.Module): Discriminator network.
        loss_function (Loss): Loss function.
        metrics (Metrics): Metrics for evaluation.
        optimizer (list): Optimizers for generator and discriminator.
        device (torch.device): The device used for training (CPU or GPU).
    """
    
    def __init__(self, **kwargs):
        super().__init__()

        self.netG, self.netD = nn.ModuleDict(), nn.ModuleDict()
        self.models = [self.netG, self.netD]
        self.ngf, self.ndf = kwargs["generator_feat"], kwargs["discriminator_feat"]
        self.ngpu += 1 if self.device.type == 'cuda' else 0
        self.in_channels = kwargs["in_channels"]
        self.init_model()
        
        self.loss_function = Loss(kwargs["functions"])
        self.metrics = Metrics(self.keys)
        self.lr = kwargs["lr"]
        self.latent_space = kwargs["latent_channels"]
        self.optimizer = [optimizer(
            model.parameters(),
            lr=self.lr,
            **{k:v for k,v in kwargs.items() if k in ["weight_decay", "momentum"]}
            ) for optimizer, model in zip(kwargs["optimizer"], self.models)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_learning(self, **kwargs):   
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(256, self.latent_space, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.

        # Setup Adam optimizers for both G and D
        #optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        #optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999)) 
        return self
       
    def init_model(self): 
        self.netG = Generator(self.ngpu, self.ngf, self.in_channels).to(self.device)
        self.netD = Discriminator(self.ngpu, self.ngf, self.in_channels).to(self.device)
        
        # Handle multi-GPU if desired
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            netG = nn.DataParallel(netG, list(range(self.ngpu)))
            netD = nn.DataParallel(netD, list(range(self.ngpu)))

        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)
        print("Generator network: ")
        print(self.netG)
        print("-"*80)
        print("Discriminator network: ")
        print(self.netD)
        print("-"*80)        
        
        
    # custom weights initialization called on ``netG`` and ``netD``
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
            
            
    def training_routine(self, epochs, train_loader, val_loader, ckpt_folder=None):
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0
        prev_errG = 1e10
        print("Starting Training Loop...")
        # For each epoch
        for epoch in epochs:
            # For each batch in the dataloader
            for i, data in enumerate(train_loader, 0):
                
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                self.netD.zero_grad()
                real_cpu = data[0].to(self.device).unsqueeze(1)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                output = self.netD(real_cpu).view(-1)
                errD_real = self.loss_function(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, self.latent_space, 1, 1, device=self.device)
                fake = self.netG(noise)
                label.fill_(self.fake_label)
                output = self.netD(fake.detach()).view(-1)
                errD_fake = self.loss_function(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                self.optimizer[1].step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(self.real_label)  
                output = self.netD(fake).view(-1)
                errG = self.loss_function(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizer[0].step()

                if i % 100 == 0:
                    print('[%d/%d][%03d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, len(epochs), i, len(train_loader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == len(epochs)-1) and (i == len(train_loader)-1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, nrow=20, padding=2, normalize=True))
                    
                    if errG.item() < prev_errG:
                        print("errG has decreased from {:.4f} to {:.4f}. Saving checkpoint...".format(prev_errG, errG))
                        prev_errG = errG.item()
                        # Save the current model state and errG value to a checkpoint file
                        checkpoint = {
                            "GAN": self.netG.state_dict(),
                            "GAN_optim": self.optimizer[0].state_dict(),
                            "epoch": epoch
                        }
                        ckpt = os.path.join(ckpt_folder,"{:03d}.pth".format(epoch))
                        ckpt = ckpt.split(".pth")[0] + "_best.pth"
                        torch.save(checkpoint, ckpt)
                    else:
                        print("errG has not decreased. Skipping checkpoint...")
                    clean_old_checkpoints(ckpt_folder)
                
                iters += 1
    
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