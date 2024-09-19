import glob
import PIL
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import pickle

types = ['scene', 'moments_dirs']
for idx in [2, 0,1]:
    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=36)

    if idx == 0:
        scene_path = '/sensei-fs-3/users/okara/camera_as_rays/RayDiffusion/output_images_4/scene/val/fig_unseen_*.png'
        # remove the ones with gt in it
        # scene_path = [path for path in scene_path if 'gt' not in path]
        paths = glob.glob(scene_path)
        paths = [path for path in paths if 'gt' not in path]
        # undo
        paths = [path for path in paths if 'undo' not in path]
        # sort paths by the number in the filename
        paths = sorted(paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        images = [Image.open(path) for path in paths]
        # on each image, writhe the epoch number in the top left corner
        for i, image in enumerate(images):
            draw = ImageDraw.Draw(image)
            draw.text((0, 0), f'Epoch: {i}', fill='black', font=font)
            del draw
        images[0].save('output_scene_val_4.gif', save_all=True, append_images=images[1:][::10], loop=0)
    elif idx == 1:
        paths = glob.glob('/sensei-fs-3/users/okara/camera_as_rays/RayDiffusion/output_images_4/moments_dirs/val/fig_unseen_*.png')
        # remove the ones that include word 'undo'
        paths = [path for path in paths if 'undo' not in path]
        # gt
        paths = [path for path in paths if 'gt' not in path]
        # sort paths by the number in the filename
        paths = sorted(paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        images = [Image.open(path) for path in paths]
        # on each image, writhe the epoch number in the top left corner
        for i, image in enumerate(images):
            draw = ImageDraw.Draw(image)
            # increase size of font
            draw.text((0, 0), f'Epoch: {i}', fill='black', font=font)
            del draw
        images[0].save('output_moment_dir_val_4.gif', save_all=True, append_images=images[1:][::10], loop=0)

    else:

        # load this
        path = '/sensei-fs-3/users/okara/camera_as_rays/RayDiffusion/output_images_4/loss_dict.pkl'
        with open(path, 'rb') as f:
            loss_dict = pickle.load(f)
        # plot the loss
        import matplotlib.pyplot as plt
        # dict is like this: {0: [train_0, val_0], 1: [train_1, val_1], ...}
        train_losses = [loss_dict[key][0] for key in loss_dict]
        val_losses = [loss_dict[key][1] for key in loss_dict]
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.savefig('loss_4.png')



