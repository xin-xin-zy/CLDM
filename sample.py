import os

from share import *

from cldm.model import create_model, load_state_dict
import cv2
from annotator.util import resize_image
import numpy as np
import torch
import einops
from cldm.ddim_hacked import DDIMSampler
from PIL import Image

# Set the paths to the input and output folders
input_folder = '  '
output_folder = ' '

# your checkpoint path
resume_path = ' ' 

ddim_steps = 50
model = create_model(' ').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# Iterate over all image files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.Bmp'):
        try:
           
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error: Failed to read image '{filename}'")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = resize_image(img, 512)

            control = torch.from_numpy(img.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(N)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            c_cat = control.cuda()
            c = model.get_unconditional_conditioning(N)
            uc_cross = model.get_unconditional_conditioning(N)
            uc_cat = c_cat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            cond={"c_concat": [c_cat], "c_crossattn": [c]}
            b, c, h, w = cond["c_concat"][0].shape
            shape = (4, h // 8, w // 8)

            samples, intermediates = ddim_sampler.sample(ddim_steps, N, 
                                                         shape, cond, verbose=False, eta=0.0, 
                                                         unconditional_guidance_scale=9.0,
                                                         unconditional_conditioning=uc_full
                                                         )
            x_samples = model.decode_first_stage(samples)
            x_samples = x_samples.squeeze(0)
            x_samples = (x_samples + 1.0) / 2.0
            x_samples = x_samples.transpose(0, 1).transpose(1, 2)
            x_samples = x_samples.cpu().numpy()
            x_samples = (x_samples * 255).astype(np.uint8)

            # Construct the file path to the output image
            output_path = os.path.join(output_folder,filename)

            Image.fromarray(x_samples).save(output_path)
        except Exception as e:
            print(f"Error processing image '{filename}': {e}")

# Release GPU resources used by the model
torch.cuda.empty_cache()

