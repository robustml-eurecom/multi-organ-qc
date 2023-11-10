#%matplotlib inline
import argparse
import os
import random
import yaml
import time
import nibabel as nib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torch import autograd
import torchvision.utils as vutils

import lpips

from models.loss import Loss
from models.metrics import Metrics
from models.CAE.cae import clean_old_checkpoints

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Generator Code
class Generator(nn.Module):
    def __init__(self, nz, ngpu, ngf, c):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            #TODO: try with a 2d latent input space with ConvTranspose2d(..., 3, 1, 0)
            nn.ConvTranspose2d(nz, ngf * 4, 3, 1, 1, bias=False),
            nn.ConvTranspose2d(ngf* 4, ngf * 8, 4, 2, 1, bias=False),
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
            nn.Sigmoid()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, ngpu, nz, ndf, c):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 256 x 256``
            nn.Conv2d(c, ndf, 4, 2, 1, bias=False),
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf=64) x 32 x 32``
            
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
            nn.LeakyReLU(0.2, inplace=True)
            # state size. ``(ndf*8=512) x 4 x 4``
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
            nn.Conv2d(ndf * 8, nz, 3, 1, 0, bias=False),
            # state size. ``(ndf*8=512) x 2 x 2``
            #TODO: try outputting a (N, 512, 2, 2) with Conv2d(..., 3, 1, 0)
        )
        self.discriminator = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def embed(self, input):
        return self.encoder(self.main(input))
    
    def forward(self, input):
        return self.discriminator(self.main(input))
    

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ngpu = 1 if self.device.type == 'cuda' else 0

        self.netG, self.netD = nn.ModuleDict(), nn.ModuleDict()
        self.ngf, self.ndf = kwargs["generator_feat"], kwargs["discriminator_feat"]
        self.in_channels = kwargs["in_channels"]
        
        self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        self.loss_function = Loss(kwargs["functions"], 0, 0)
        self.lr = kwargs["lr"]
        self.latent_space = kwargs["latent_channels"]
        
        self.init_model()
        self.models = [self.netG, self.netD]
        #print(self.models)
        self.optimizer = [
            torch.optim.Adam(
                model.parameters(),
                lr=self.lr,
                betas=(kwargs["beta1"], 0.999)
            ) for model in self.models]
        self.setup_learning()
        
    def setup_learning(self):   
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(256, self.latent_space, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.
        return self
       
    def init_model(self): 
        self.netG = Generator(self.latent_space, self.ngpu, self.ngf, self.in_channels).to(self.device)
        self.netD = Discriminator(self.ngpu, self.latent_space, self.ndf, self.in_channels).to(self.device)
        
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
        return self    
        
        
    # custom weights initialization called on ``netG`` and ``netD``
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    
    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty
        
            
    def training_routine(self, epochs, train_loader, val_loader, ckpt_folder=None):
        img_list = []
        G_losses, D_losses= [], []
        recent_losses, ema_loss = [], []
        iters = 0
        prev_errG = float('inf')
        
        print("Starting Training Loop...")
        # For each epoch
        start_time = time.time()
        for epoch in epochs:
            # For each batch in the dataloader
            for i, data in enumerate(train_loader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                self.netD.zero_grad()
                real_cpu = data[0].unsqueeze(0).float().to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                output = self.netD(real_cpu).view(-1)
                errD_real = self.loss_function(output, label, epoch)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(b_size, self.latent_space, 4, 4, device=self.device)
                noise = self.netD.embed(real_cpu) # (N, 100, 4, 4)
                fake = self.netG(noise)
                #print(noise.shape, fake.shape)
                label.fill_(self.fake_label)
                output = self.netD(fake.detach()).view(-1)
                errD_fake = self.loss_function(output, label, epoch)
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
                errG = .1 * self.loss_function(output, label, epoch) + .9 * (.5 * nn.MSELoss()(fake, real_cpu) + .5 * self.lpips_fn(fake, real_cpu))
                errG.backward()
                D_G_z2 = output.mean().item()
                self.optimizer[0].step()

                if i % 100 == 0:
                    print('[%03d/%03d][%03d/%03d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch + 1, len(epochs), i, len(train_loader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                recent_losses.append(errG.item())

                # Calculate the mean of recent losses
                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % len(train_loader) == 0) or ((epoch == len(epochs)-1) and (i == len(train_loader)-1)):
                    with torch.no_grad():
                        if val_loader is not None: val_data = torch.cat([inputs for inputs in val_loader], dim=0).float().to(self.device)
                        emb = self.netD.embed(val_data if val_loader is not None else self.fixed_noise)
                        fake = self.netG(emb if val_loader is not None else self.fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, normalize=True))
                iters += 1
 
            print("Time elapsed: {:.2f} seconds".format(time.time() - start_time))
            avg_loss = sum(recent_losses) / len(recent_losses) #if epoch == 0 else .9 * sum(recent_losses) / len(recent_losses) + .1 * avg_loss
            ema_loss.append(avg_loss)
            
            if avg_loss < prev_errG:
                print("Generator Loss has decreased from {:.4f} to {:.4f}. Saving checkpoint...".format(prev_errG, avg_loss))
                prev_errG = avg_loss
                checkpoint = {
                    "dcgan_g": self.netG.state_dict(),
                    "dcgan_d": self.netD.state_dict(),
                    "dcgan_g_optim": self.optimizer[0].state_dict(),
                    "dcgan_d_optim": self.optimizer[1].state_dict(),
                    "epoch": epoch
                }
                if not os.path.exists(ckpt_folder): os.makedirs(ckpt_folder)
                ckpt = os.path.join(ckpt_folder, "{:03d}.pth".format(epoch))
                ckpt = ckpt.split(".pth")[0] + "_best.pth"
                torch.save(checkpoint, ckpt)
            else:
                print("Generator error has not decreased. Skipping checkpoint...")
            clean_old_checkpoints(ckpt_folder)
            recent_losses = []
                
        return ([G_losses, D_losses, ema_loss], img_list)
    
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
            ckpt = os.path.join(data_path,"checkpoints/dcgan", sorted([file for file in os.listdir(os.path.join(data_path,"checkpoints/dcgan")) if "_best" in file])[-1])
        else:
            ckpt = checkpoint_path
        print("Chosen checkpoint is {} .".format(os.path.split(ckpt)[1]))
        print("###################################")
        ckpt = torch.load(ckpt)

        self.netG.load_state_dict(ckpt["dcgan_g"])
        self.netD.load_state_dict(ckpt["dcgan_d"])
        self.optimizer[0].load_state_dict(ckpt["dcgan_g_optim"])
        self.optimizer[1].load_state_dict(ckpt["dcgan_d_optim"])
        if eval: self.netG.eval(), self.netD.eval()