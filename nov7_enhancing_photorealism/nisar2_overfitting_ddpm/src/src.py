import torch
from torch.optim import Adam
from torchvision import transforms
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
import numpy as np
from natsort import natsorted
import imageio
import glob

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the dataset and load all data in memory
class HypersimDataset:
    def __init__(self, root_dir, scenes_cams_frames_pth, transform=None):
        self.root_dir = root_dir
        self.scenes_cams_frames = pd.read_csv(scenes_cams_frames_pth, dtype={'frame': str})
        self.transform = transform
        self.data = self.get_all_frame_data()  # Load all data into memory

    def get_all_frame_data(self):
        all_data = []
        for _, row in self.scenes_cams_frames.iterrows():
            frame_data = self.load_frame_data(row)
            all_data.append(frame_data)
        return all_data

    def load_frame_data(self, scene_cam_frame):
        # Load image paths and apply transformations
        scene, cam, frame = scene_cam_frame['scene'], scene_cam_frame['cam'], scene_cam_frame['frame']
        images_pth = os.path.join(self.root_dir, scene, 'images')
        final_preview_pth = os.path.join(images_pth, f'scene_{cam}_final_preview')
        
        # Load the target image and conditioning images
        def load_image(path):
            img = Image.open(path)
            return self.transform(img) if self.transform else img

        frame_data = {
            'final_color': load_image(os.path.join(final_preview_pth, f'frame.{frame}.color.jpg')),
            'final_diffuse_illumination': load_image(os.path.join(final_preview_pth, f'frame.{frame}.diffuse_illumination.jpg')),
            'final_diffuse_reflectance': load_image(os.path.join(final_preview_pth, f'frame.{frame}.diffuse_reflectance.jpg')),
            'final_gamma': load_image(os.path.join(final_preview_pth, f'frame.{frame}.gamma.jpg')),
            'final_lambertian': load_image(os.path.join(final_preview_pth, f'frame.{frame}.lambertian.jpg')),
            'final_non_lambertian': load_image(os.path.join(final_preview_pth, f'frame.{frame}.non_lambertian.jpg')),
            'final_tonemap': load_image(os.path.join(final_preview_pth, f'frame.{frame}.tonemap.jpg')),
            'geo_gamma': load_image(os.path.join(images_pth, f'scene_{cam}_geometry_preview', f'frame.{frame}.gamma.jpg')),
            'geo_normal_bump_world': load_image(os.path.join(images_pth, f'scene_{cam}_geometry_preview', f'frame.{frame}.normal_bump_world.png'))[:3, :, :],
            'geo_tex_coord': load_image(os.path.join(images_pth, f'scene_{cam}_geometry_preview', f'frame.{frame}.tex_coord.png'))[:3, :, :]
        }
        return frame_data

# Define model
class DownsampledUNet2DModel(UNet2DModel):
    def __init__(self, original_in_channels, downsampled_channels=8, **kwargs):
        super().__init__(in_channels=downsampled_channels, **kwargs)
        self.downsample_conv = torch.nn.Conv2d(original_in_channels, downsampled_channels, kernel_size=1)
        
    def forward(self, sample, timestep, conditioning_images=None):
        condition = torch.cat([conditioning_images[key] for key in conditioning_images], dim=1)
        sample = torch.cat([sample, condition], dim=1)
        sample = self.downsample_conv(sample)
        return super().forward(sample, timestep).sample


def run(root_dir, scenes_cams_frames_pth, output_pth, num_epochs=2001):
    os.makedirs(output_pth, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = HypersimDataset(root_dir=root_dir, scenes_cams_frames_pth=scenes_cams_frames_pth, transform=transform)

    output_im_pth = os.path.join(output_pth, 'ims')
    os.makedirs(output_im_pth, exist_ok=True)
    model_checkpoint_pth = os.path.join(output_pth, 'checkpoints')
    os.makedirs(model_checkpoint_pth, exist_ok=True)

    # Initialize model, scheduler, and optimizer
    num_conditioning_images = len(dataset.data[0]) - 1  # Exclude target image
    total_in_channels = 3 + 3 * num_conditioning_images

    custom_unet = DownsampledUNet2DModel(
        original_in_channels=total_in_channels,
        downsampled_channels=8,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to('cuda')

    scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear")
    optimizer = Adam(custom_unet.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(num_epochs):
        for frame_data in dataset.data:
            # Separate target and conditioning images
            target_image = frame_data['final_color'].unsqueeze(0).to('cuda')
            conditioning_images = {k: v.unsqueeze(0).to('cuda') for k, v in frame_data.items() if k != 'final_color'}
            
            target_image = (target_image * 2) - 1  # Normalize to [-1, 1]

            # Sample random timestep
            timestep = torch.randint(0, scheduler.num_train_timesteps, (1,), device='cuda').long()

            # Generate noise and create noisy image
            noise = torch.randn_like(target_image, device='cuda')
            noisy_target = scheduler.add_noise(target_image, noise, timestep)

            # Forward pass
            output = custom_unet(noisy_target, timestep, conditioning_images=conditioning_images)

            # Compute loss between predicted noise and actual noise
            loss = torch.nn.functional.mse_loss(output, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

        # Periodic inference visualization
        if epoch % 100 == 0 and epoch > 0:
            custom_unet.eval()
            generated_image = torch.randn_like(target_image).to('cuda')
            num_inference_steps = 1000
            scheduler.set_timesteps(num_inference_steps=num_inference_steps)
            with torch.no_grad():
                for t in tqdm(scheduler.timesteps, leave=False):
                    model_output = custom_unet(generated_image, t, conditioning_images=conditioning_images)
                    generated_image = scheduler.step(model_output, t, sample=generated_image).prev_sample
            generated_image_normalized = (generated_image + 1) / 2
            plt.imshow(generated_image_normalized.squeeze().permute(1, 2, 0).cpu().detach().numpy())
            plt.title(f'Generated Color at Epoch {epoch}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_im_pth, f'generated_color_{epoch}.png'))
            custom_unet.train()

            # Save model checkpoint
            checkpoint_path = os.path.join(model_checkpoint_pth, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': custom_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"Saved checkpoint at {checkpoint_path}")


def gif_maker(root_dir, scenes_cams_frames_pth, output_pth, chekpoint_pth, idx=0):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = HypersimDataset(root_dir=root_dir, scenes_cams_frames_pth=scenes_cams_frames_pth, transform=transform)
    num_conditioning_images = len(dataset.data[0]) - 1  # Exclude target image
    total_in_channels = 3 + 3 * num_conditioning_images
    custom_unet = DownsampledUNet2DModel(
            original_in_channels=total_in_channels,
            downsampled_channels=8,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        ).to('cuda')

    checkpoint_pth = chekpoint_pth
    print(checkpoint_pth)
    checkpoint = torch.load(checkpoint_pth)
    custom_unet.load_state_dict(checkpoint['model_state_dict'])
    custom_unet.eval()


    frame_data = dataset.data[idx]
    target_image = frame_data['final_color'].unsqueeze(0).to('cuda')
    conditioning_images = {k: v.unsqueeze(0).to('cuda') for k, v in frame_data.items() if k != 'final_color'}
    generated_image = torch.randn_like(target_image).to('cuda')

    scheduler = DDPMScheduler(beta_start=0.0001, beta_end=0.02, beta_schedule="linear")
    num_inference_steps = 1000
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    denoised_images_output_pth = os.path.join(output_pth, f'denoised_ims{idx}')
    os.makedirs(denoised_images_output_pth, exist_ok=True)

    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, leave=False):
            model_output = custom_unet(generated_image, t, conditioning_images=conditioning_images)
            generated_image = scheduler.step(model_output, t, sample=generated_image).prev_sample
            generated_image_normalized = (generated_image + 1) / 2
            plt.imshow(generated_image_normalized.squeeze().permute(1, 2, 0).cpu().detach().numpy())
            plt.title(f'Generated Color at Timestep {t}')
            plt.tight_layout()
            plt.savefig(os.path.join(denoised_images_output_pth, f'generated_color_{t}.png'))
            plt.close()
    custom_unet.train()
    images_for_gif = []

    denoised_img_pths = list(reversed(natsorted(os.listdir(denoised_images_output_pth))))
    for i in range(0,len(denoised_img_pths),10):
        img_path = f'{denoised_images_output_pth}/{denoised_img_pths[i]}'

        images_for_gif.append(imageio.v2.imread(img_path))


    gif_output_path = f'{denoised_images_output_pth}/animation.gif'
    imageio.mimsave(gif_output_path, images_for_gif, duration=1/240, loop=0)

def extract_losses_from_checkpoints(checkpoint_dir):
    # List to store the extracted data
    losses_data = []

    # Find all checkpoint files in the directory
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')))

    for checkpoint_path in checkpoint_files:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Extract epoch and loss
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        # Append to data list
        losses_data.append({'epoch': epoch, 'loss': loss})
        print(f"Loaded checkpoint {checkpoint_path}: Epoch {epoch}, Loss {loss}")

    # Convert to a DataFrame for analysis
    losses_df = pd.DataFrame(losses_data)
    losses_df = losses_df.sort_values(by='epoch')
    return losses_df