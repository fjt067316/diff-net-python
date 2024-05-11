import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import random


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = [filename for filename in os.listdir(root_dir) if filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            # Open the image
            image = Image.open(img_path)

            # Convert RGBA images to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Apply transformations if specified
            if self.transform:
                image = self.transform(image)
            # print(f"got {img_path}")
            return image
        except Exception as e:
            print(f"Error opening image '{img_path}': {e}")
            random_idx = random.randint(0, self.__len__() - 1)
            return self.__getitem__(idx)


# Define transformations for your images
transform = transforms.Compose([
    # transforms.Resize((1024, 1536)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # Convert image to tensor
    # transforms.Lambda(grey_scale_to_rgb),  # Convert grayscale to RGB
    transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize image
])

# Define the path to the folder containing your images
root_dir = './images/'

# Create a custom dataset
dataset = CustomDataset(root_dir, transform=transform)
print(dataset.__len__())

import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 800
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

from torch import nn
import math


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1, bias=False)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)

# print("Num params: ", sum(p.numel() for p in model.parameters()))
# model
    
def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

import numpy as np

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    plt.imshow(reverse_transforms(image))

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img = torch.randn((1, 3, 512, 768), device=device)
    plt.figure(figsize=(40,30))
    plt.axis('off')
    num_images = 20
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img[0].detach().cpu())
    plt.show()

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
from torch.cuda.amp import autocast, GradScaler

# Enable CuDNN benchmarking
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleUnet()
model.to(device)

try:
    model.load_state_dict(torch.load('unet.pth'))
except:
    pass


optimizer = AdamW(model.parameters(), lr=0.00005)

try:
    optimizer.load_state_dict(torch.load('adamw.pth'))
except FileNotFoundError:
    pass

scaler = GradScaler()
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
BATCH_SIZE = 12 # 16024MiB /  16380MiB 
gradient_accumulation_steps = 4

# Define learning rate boundaries
lr_min = 0.000001
lr_max = 0.0005

# Define step size (number of iterations in half a cycle) keep in mind grad accumulation defers the step
step_size_up = 100
step_size_down = 100

# Create cyclic learning rate scheduler
scheduler = CyclicLR(optimizer, base_lr=lr_min, max_lr=lr_max,
                     step_size_up=step_size_up, step_size_down=step_size_down, mode='triangular', cycle_momentum=False)

epochs = 400
max_loss = 0

torch.cuda.empty_cache()

for epoch in range(epochs):

    # if (epoch+1) % 10 == 0: # reshuffle and get new data
    #   del dataset
    #   dataset = CustomDataset(root_dir, transform=transform)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) # iterate over dataset multiple times
    optimizer.zero_grad()
    print(f"Epoch {epoch}/{epochs} max_loss {max_loss}")
    max_loss = 0

    for step, batch in enumerate(dataloader):
        if batch == None or batch.shape[0] != BATCH_SIZE:
          print(f"batch of step {step} is not full or is None")
          continue
        with autocast():
            # t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            t = torch.tensor([torch.distributions.exponential.Exponential(0.005).sample() for _ in range(BATCH_SIZE)])
            t = (t % T).long().to(device)
            # print(t)

            loss = get_loss(model, batch.to(device), t)

        print(f"step: {step} loss: {loss}")
        max_loss = max(loss, max_loss)

        # loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

        # gradient accumulation
        # if (step + 1) % gradient_accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # Update learning rate
        optimizer.zero_grad()

        if step % 250 == 0:
            # sample_plot_image()
            torch.save(model.state_dict(), f'saves/unet_{epoch}_{step/100}.pth')
            torch.save(optimizer.state_dict(), f'saves/adamw_{epoch}_{step/100}.pth')