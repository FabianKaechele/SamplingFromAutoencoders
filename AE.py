#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from Sampling import *

import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
robjects.numpy2ri.activate()


# In[ ]:


class AE_MNIST(nn.Module):
    def __init__(self, image_size, channel_num, kernel_num, z_size, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder
        self.encoder = nn.Sequential(
            _conv(channel_num, kernel_num // 4),
            _conv(kernel_num // 4, kernel_num // 2),
            _conv(kernel_num // 2, kernel_num),
        )

        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # decoder
        self.decoder = nn.Sequential(
            _deconv(kernel_num, kernel_num // 2),
            _deconv(kernel_num // 2, kernel_num // 4),
            _deconv(kernel_num // 4, channel_num),
            nn.Sigmoid()
        )

        # projection
        self.project = _linear(z_size, self.feature_volume, relu=False)
        self.q_layer = _linear(self.feature_volume, z_size, relu=False)

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_layer(unrolled)
    
    def encode(self,x):
        unrolled = self.encoder(x).view(-1, self.feature_volume)
        z = self.q_layer(unrolled)
        return z
    
    def decode(self,z):
        x_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        x_reconstructed = self.decoder(x_projected)
        return x_reconstructed
        
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
    def vine_sample(self, vine, size, noise=None):
        rvinecop = importr('rvinecopulib')
        if noise is None:
            sampled_r = rvinecop.rvine(size, vine)
        else:
            sampled_r = rvinecop.inverse_rosenblatt(noise, vine)
        sampled_py = torch.Tensor(np.asarray(sampled_r)).view(size, -1).to(self.device)
        del sampled_r
        img_new = self.decode(sampled_py)
        return img_new
    
    def beta_sample1(self,x,y,size,seed=500):
        y_sample = sampling1(x,y, size, seed=500)
   
        img_new = self.decode(torch.tensor(y_sample).float())
        return img_new


# In[ ]:


class VAE_MNIST(nn.Module):
    def __init__(self, image_size, channel_num, kernel_num, z_size, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder
        self.encoder = nn.Sequential(
            _conv(channel_num, kernel_num // 4),
            _conv(kernel_num // 4, kernel_num // 2),
            _conv(kernel_num // 2, kernel_num),
        )

        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # q
        self.q_mean = _linear(self.feature_volume, z_size, relu=False)
        self.q_logvar = _linear(self.feature_volume, z_size, relu=False)

        # projection
        self.project = _linear(z_size, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            _deconv(kernel_num, kernel_num // 2),
            _deconv(kernel_num // 2, kernel_num // 4),
            _deconv(kernel_num // 4, channel_num),
            nn.Sigmoid()
        )
    def encode(self,x):
        encoded = self.encoder(x)
        mean, logvar = self.q(encoded)
        return mean, logvar
    
    def decode(self,z):
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,)
        x_reconstructed = self.decoder(z_projected)
        return x_reconstructed        

    
#     def forward(self, x):
#         # encode x
#         encoded = self.encoder(x)

#         # sample latent code z from q given x.
#         mean, logvar = self.q(encoded)
#         z = self.z(mean, logvar)
#         z_projected = self.project(z).view(
#             -1, self.kernel_num,
#             self.feature_size,
#             self.feature_size,
#         )

#         # reconstruct x from z
#         x_reconstructed = self.decoder(z_projected)

#         # return the parameters of distribution of q given x and the
#         # reconstructed image.
#         return (mean, logvar), x_reconstructed
    
    def forward(self, x):
        mean, logvar = self.encode(x)      
        z = self.z(mean, logvar)
        x_reconstructed = self.decode(z)
        return (mean, logvar), x_reconstructed

    # ==============
    # VAE components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(mean.shape).to(self.device)
        return eps.mul(std).add_(mean)

    def kl_divergence_loss(self, mean, logvar):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()
    
    def sample(self, size, noise=None):
        if noise is None:
            z = Variable(torch.randn(size, self.z_size)).to(self.device)
        else:
            z = noise
        return self.decode(z)

# In[ ]:


class AE_Celeba(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = 64
        self.hidden_dim = 100
        self.z_size = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channel_num = 3

        # Decoder - Fractional strided convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 0, bias=False), #nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()  # nn.Tanh()
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

        # Fully-connected layers
        self.fc1 = nn.Linear(512, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.z_size)
        self.fc3 = nn.Linear(self.z_size, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 512)

    def encode(self,x):
        encoded = F.relu(self.fc1(self.encoder(x).view(x.size(0), -1)))
        return self.fc2(encoded)
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = torch.sigmoid(self.fc4(h3))
        return self.decoder(h4.view(z.size(0), -1, 1, 1))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
    def vine_sample(self, vine, size, noise=None):
        rvinecop = importr('rvinecopulib')
        if noise is None:
            sampled_r = rvinecop.rvine(size, vine)
        else:
            sampled_r = rvinecop.inverse_rosenblatt(noise, vine)
        sampled_py = torch.Tensor(np.asarray(sampled_r)).view(size, -1).to(self.device)
        del sampled_r
        img_new = self.decode(sampled_py)
        return img_new
    
    def beta_sample1(self,x,y,size,seed=500):
        y_sample = sampling1(x,y, size, seed=500)
        img_new = self.decode(torch.tensor(y_sample).float())
        return img_new
    



# In[ ]:

class VAE_Celeba(nn.Module):

    def __init__(self, image_size, channel_num, kernel_num, z_size, **kwargs):
        super(VAE_Celeba, self).__init__()
     
        self.image_size = 64
        self.model_name='VAE_Celeba'
        self.hidden_dim = 100
        self.z_size = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channel_num = 3

        # Decoder - Fractional strided convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()  # nn.Tanh()
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

        # Fully-connected layers
        self.fc1 = nn.Linear(512, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.z_size)
        self.fc22 = nn.Linear(self.hidden_dim, self.z_size)
        self.fc3 = nn.Linear(self.z_size, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 512)
        
    def encode(self,x):
        encoded = F.relu(self.fc1(self.encoder(x).view(x.size(0), -1)))
        mu = self.fc21(encoded)
        logvar = self.fc22(encoded)
        return mu,logvar

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.sigmoid(self.fc4(h3))
        return self.decoder(h4.view(z.size(0), -1, 1, 1))
    
    def reparametrize(self,mu,logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z

    def forward(self, x):
        mu,logvar = self.encode(x)
        z = self.reparametrize(mu,logvar)
        decoded = self.decode(z)
        return (mu, logvar), decoded

    def kl_divergence_loss(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    def sample(self, size, noise=None):
        if noise is None:
            z = Variable(torch.randn(size, self.z_size)).to(self.device)
        else:
            z = noise
        return self.decode(z)



   
    
class ae_SVHN(nn.Module):

    def __init__(self, image_size, hidden_dim, z_size, device, channel_num=3):
        super(ae_SVHN, self).__init__()
        self.encoding_dim = 20
        self.image_size = 32
        self.hidden_dim = 20
        self.model_name = "ae_SVHN"
        self.z_size = 32
        self.device = device
        self.channel_num = 3

        # Decoder - Fractional strided convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()  # nn.Tanh()
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

        # Fully-connected layers
        self.fc1 = nn.Linear(256, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc3 = nn.Linear(self.encoding_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 256)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.sigmoid(self.fc4(h3))
        return self.decoder(h4.view(z.size(0), -1, 1, 1))
    
    def encode(self,x):
        encoded = F.relu(self.fc1(self.encoder(x).view(x.size(0), -1)))
        return self.fc21(encoded)

    def forward(self, x):
        # Encode
        encoded = F.relu(self.fc1(self.encoder(x).view(x.size(0), -1)))

        e = self.fc21(encoded)

        # Decode
        decoded = self.decode(e)

        return decoded

    def sample(self, size, vine, noise=None):

            if noise is None:
                sampled_r = rvinecop.rvine(size, vine)
            else:
                sampled_r = rvinecop.inverse_rosenblatt(noise.cpu().numpy(), vine)

            sampled_py = torch.Tensor(np.asarray(sampled_r)).view(size, -1).to(self.device)

            # Decode
            decoded = self.decode(sampled_py)

            del sampled_py

            return decoded

    @property
    def name(self):
        return (
            'ae_SVHN'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )


class vae_SVHN(nn.Module):

    def __init__(self, image_size, hidden_dim, z_size, device, channel_num=3 ):
        super(vae_SVHN, self).__init__()

        self.label = "vae_SVHN"
        self.encoding_dim = 20
        self.image_size = 32
        self.hidden_dim = 20
        self.model_name = "vae_SVHN"
        self.z_size = 20
        self.device = device
        self.channel_num = 3

        # Decoder - Fractional strided convolutional layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()  # nn.Tanh()
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

        # Fully-connected layers
        self.fc1 = nn.Linear(256, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.encoding_dim)
        self.fc3 = nn.Linear(self.encoding_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, 256)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.sigmoid(self.fc4(h3))
        return self.decoder(h4.view(z.size(0), -1, 1, 1))
    
    def encode(self,x):
        
         # Encode
        encoded = F.relu(self.fc1(self.encoder(x).view(x.size(0), -1)))

        # Obtain mu and logvar
        mu = self.fc21(encoded)
        logvar = self.fc22(encoded)
        return mu,logvar
        

    def forward(self, x):
        # Encode
        encoded = F.relu(self.fc1(self.encoder(x).view(x.size(0), -1)))

        # Obtain mu and logvar
        mu = self.fc21(encoded)
        logvar = self.fc22(encoded)

        # Reparametrization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        # Decode
        decoded = self.decode(z)

        # return decoded, mu, logvar
        return (mu, logvar), decoded

    def kl_divergence_loss(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    def sample(self, size, noise=None):
        if noise is None:
            z = Variable(torch.randn(size, self.z_size)).to(self.device)
        else:
            z = noise
        return self.decode(z)

    @property
    def name(self):
        return (
            'vae_SVHN'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )



def _conv(channel_size, kernel_num):
    return nn.Sequential(
        nn.Conv2d(
            channel_size, kernel_num,
            kernel_size=4, stride=2, padding=1,
        ),
        nn.BatchNorm2d(kernel_num),
        nn.ReLU(),
    )
def _deconv(channel_num, kernel_num):
    return nn.Sequential(
        nn.ConvTranspose2d(
            channel_num, kernel_num,
            kernel_size=4, stride=2, padding=1,
        ),
        nn.BatchNorm2d(kernel_num),
        nn.ReLU(),
    )

def _linear(in_size, out_size, relu=True):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.ReLU(),
    ) if relu else nn.Linear(in_size, out_size)


def get_noise(noise_num=64, latent=100):
     return Variable(torch.randn((noise_num, latent, 1, 1)))