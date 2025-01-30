
import os
os.environ["CUDA_VISIBLE_DEVICES"]='5'
from diffusers import StableDiffusionPipeline,EulerDiscreteScheduler,DDIMScheduler,EulerAncestralDiscreteScheduler,DDPMScheduler
# from make_captions import make_captions

import torch
from math import sqrt
image_num=2


def create_images(prompts,output_path,lora_dir):

    model_id = "runwayml/stable-diffusion-v1-5"
    model_id = "stabilityai/stable-diffusion-2-base"
    # model_id = "/public/zhouchao/stable-diffusion-2/"
    model_id = "/home/czhou/data/sd-1.5"
    # model_id = "/public/zhouchao/stable-diffusion-2-1/"
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    diffuser = StableDiffusionPipeline.from_pretrained(model_id,
                                                   scheduler = scheduler,
                                                    safety_checker = None,
                                                    # requires_safety_checker = False
                                                   ).to(f"cuda:0")

    generator = torch.manual_seed(40)   
    images = diffuser(prompts,
                      negative_prompt=f"worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, vague",
                      generator = generator,   
                      num_inference_steps=50, 
                      guidance_scale=7.5,
                      height=512,width=512,
                      num_images_per_prompt=image_num,                                                                                                                       
                      ).images
    use_lora=False
    false_path = os.path.join(output_path,"False")
    if not os.path.exists(false_path):
        os.makedirs(false_path)
    for i,image in enumerate(images):
        image.save(os.path.join(false_path,f'{i}_{use_lora}.jpg'))
    use_lora=True

    style = os.getenv("STYLE","None")
    diffuser.unet.load_attn_procs(lora_dir)
    generator = torch.manual_seed(40)
    images = diffuser(prompts,
                      negative_prompt=f"worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting, vague",
                      generator = generator,   
                      num_inference_steps=50, 
                      guidance_scale=7.5,
                      height=512,width=512,
                      num_images_per_prompt=image_num,                                                                                                                       
                      ).images
  
    true_path = os.path.join(output_path,"True")
    if not os.path.exists(true_path):
        os.makedirs(true_path)
    for i,image in enumerate(images):
        image.save(os.path.join(true_path,f'{i}_{use_lora}.jpg'))


imageries = [
    "vase of flowers",
    "bowl of fruit",
    "still life with candles",
    "landscape with rolling hills",
    "cityscape with buildings",
    "forest with sunlight filtering through trees",
    "portrait of a person",
    "quiet beach at sunset",
    "mountain range with snow",
    "tranquil lake with reflections",
    "barn in a rural setting",
    "bustling street market",
    "boat on a calm river",
    "group of animals in a field",
    "crowded café scene",
    "horse grazing in a pasture",
    "vintage clock on a mantelpiece",
    "window with a view of the countryside",
    "room with antique furniture",
    "close-up of a tree's bark and leaves"
             ]
if __name__ == '__main__':
    import sys
    import os

    style_raw = "Vincent van Gogh"

    
    style = style_raw.replace(" ","_")
    lora_dir = f"/home/czhou/Cplug-in/data/Lora_model_for_merge/Vincent_van_Gogh_positive/lr_0.0001_0.0001_epochs_20/20"
    
    data_dir = "/home/czhou//Cplug-in/data/target_style/"

    output_dir = data_dir+f'{style}'
    surrounding_style = "Edgar Degas"
    surrounding_style = "Kazimir Malevich"
    surrounding_style = "Vincent van Gogh"
    for i,imagery in enumerate(imageries):
        os.makedirs(f"{output_dir}/{i}",exist_ok=True)
        create_images(f"{imagery} by {surrounding_style}",f"{output_dir}/{i}",lora_dir)
    