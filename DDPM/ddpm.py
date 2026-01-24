from email.mime import image
import time
from tkinter import image_names
import test
import torch
import deepinv
from torchvision import datasets, transforms

# Check for MPS (Apple Silicon) availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
batch_size = 32
image_size = 32

#transforms 
transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,)),
    ]
)

#loaders
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="./data", train=True, download=True, transform=transforms),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root="./data", train=False, download=True, transform=transforms),
    batch_size=batch_size,
    shuffle=False,
)

#learning rate
lr = 1e-4
epochs = 100

model = deepinv.models.DiffUNet(in_channels=1, out_channels=1).to(device)

#learning rate optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#loss funciton
mse = deepinv.loss.MSE()

#beta
beta_start = 1e-4
beta_end = 0.02
timesteps = 1000 

betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


for epoch in range(epochs): 
    model.train()
    for data, _ in train_loader: 
        imgs = data.to(device)
        noise = torch.rand_like(imgs)
        t = torch.randint(0, timesteps, (imgs.size(0),), device=device)
        
        noised_imgs = (sqrt_alphas_cumprod[t, None, None, None] * imgs + sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise)
        
        optimizer.zero_grad()
        estimated_noise = model(noised_imgs, t, type_t="timestep")
        loss = mse(estimated_noise, noise)
        loss.backward()
        optimizer.step()
        
        torch.save(model.state_dict(), "ddpm_mnist.pth")