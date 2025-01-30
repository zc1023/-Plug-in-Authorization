# import diffusers
# from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
# from diffusers.loaders import AttnProcsLayers
# from diffusers.models.attention_processor import LoRAAttnProcessor
# from diffusers.optimization import get_scheduler
# from diffusers.utils import check_min_version, is_wandb_available
# from diffusers.utils.import_utils import is_xformers_available

import os
os.environ["CUDA_VISIBLE_DEVICES"]='6'
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import copy
from diffusers import StableDiffusionPipeline,EulerDiscreteScheduler,DDIMScheduler,EulerAncestralDiscreteScheduler,DDPMScheduler
# from StableDiffuser import StableDiffuser
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from tqdm import tqdm
import random
import numpy as np
from math import sqrt
import sys

data_dir = "/home/czhou/Cplug-in/data/specific_prompt/"
print(sys.argv)

sa_lr = 5e-5
ca_lr = 1e-4

batchsize = 1
sample_images=4

# imageries = [
    # "vase of flowers",
    # "bowl of fruit",
    # "still life with candles",
    # "landscape with rolling hills",
    # "cityscape with buildings",
    # "forest with sunlight filtering through trees",
    # "portrait of a person",
    # "quiet beach at sunset",
    # "mountain range with snow",
    # "tranquil lake with reflections",
    # "barn in a rural setting",
    # "bustling street market",
    # "boat on a calm river",
    # "group of animals in a field",
    # "crowded café scene",
    # "horse grazing in a pasture",
    # "vintage clock on a mantelpiece",
    # "window with a view of the countryside",
    # "room with antique furniture",
    # "close-up of a tree's bark and leaves"
            #  ]
# style = "Picasso"
# imagery = sys.argv[1]
style = "Vincent Van Gogh"

resolution = 512
epochs = 10
iterations = 20
nsteps = 50
max_grad_norm = 1.0
rank = 80

import re
def extract_key(key):
    key = key.replace('.processor','')
    dot_digitals = re.findall('\.\d',key)
    for dot_digital in dot_digitals:
        digital = re.search('\d',dot_digital).group()
        target = f'[{digital}]'
        key = key.replace(dot_digital,target)
    return key

def seed_torch(seed=1029):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch(0)

train_transforms = transforms.Compose(
        [   transforms.ToPILImage(),
         
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
         
            transforms.Resize((resolution,resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5,], [0.5]),
        ]
    )
'''
the dataset need to be removed
'''
class ImgCaptionSet(torch.utils.data.Dataset):
    
    def __init__(self,root_dir,transforms=train_transforms):
        self.root_dir = root_dir
        self.transforms = transforms
        files = os.listdir(self.root_dir)
        self.images = []
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                self.images.append(file)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir,image_index)
        name,suffix = os.path.splitext(image_index)
        text_path = os.path.join(self.root_dir,f'{name}.txt')
        img = plt.imread(img_path)
        img = self.transforms(img)
        try:
            with open(text_path,'r')as f:
                caption = f.read()
        except:
            imagery = os.getenv("IMAGERY","simple content")
            caption = f'the painitng of {imagery}, high definition, high quality'
        return img,caption

def create_data(diffuser,data_path,seed=-1,type = "positive",batch_size = 2):
    diffuser.to("cuda:0")
    with torch.no_grad():
        import random
        random.seed(seed)
        seed = random.randint(1,1e5)
        generator = torch.manual_seed(seed)
        diffuser.unet.set_default_attn_processor()
        images = []
        imagery = os.getenv("IMAGERY","simple content")
        for iteration in range(sample_images//batch_size):
            if type == "positive":    
                images1 = diffuser(prompt = f"the painitng of {imagery} by a generic artist, high definition, high quality",
                                    # negative_prompt=f"by {style}",
                                    generator = generator,   
                                    num_inference_steps=50, 
                                    guidance_scale=7.5, # the quality of the generated image is low  
                                    height=resolution,width=resolution,
                                    num_images_per_prompt=batch_size,                                                                                                                       
                                ).images
                images = images+images1
            if type == "negative":
                images1 = diffuser(prompt = f"the painitng of {imagery} by {style}, high definition, high quality",
                                    generator = generator,   
                                    num_inference_steps=50, 
                                    guidance_scale=7.5,
                                    height=resolution,width=resolution,
                                    num_images_per_prompt=batch_size,                                                                                                                       
                                ).images
                images = images+images1
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    for i,image in enumerate(images):
        image.save(os.path.join(data_path,f'{i}.jpg'))
    
    diffuser.to("cpu")
    torch.cuda.empty_cache()


def train_lora(type):
    model_id = "runwayml/stable-diffusion-v1-5"
    model_id = "stabilityai/stable-diffusion-2-base"
    model_id = "/public/zhouchao/stable-diffusion-2/"
    model_id = "/home/czhou/data/sd-1.5"
    # model_id = "/public/zhouchao/stable-diffusion-2-1/"
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    diffuser = StableDiffusionPipeline.from_pretrained(model_id,
                                                   scheduler = scheduler,
                                                    safety_checker = None,
                                                    # requires_safety_checker = False
                                                   ).to("cuda:0")
    # freeze diffuser
    for model in [diffuser.unet,diffuser.vae,diffuser.text_encoder,]:
        model.requires_grad_(False)
    freezed = copy.deepcopy(diffuser)

    data_path = data_dir+f'/extract-concept/{style.replace(" ","_")}_{type}'
    output_root_dir=data_dir+f'Lora_model_for_merge/{os.path.basename(data_path)}/lr_{sa_lr}_{ca_lr}_epochs_{epochs}'
    os.makedirs(data_path,exist_ok=True)
    os.makedirs(output_root_dir,exist_ok=True)

    # create_data(freezed,data_path=data_path,type=type)

    unet = diffuser.unet
    if type == "negative":
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,rank = rank)

        # set lora
        unet.set_attn_processor(lora_attn_procs)
    elif type == "positive":
        dir=data_dir+f'Lora_model_for_merge/{style.replace(" ","_")}_negative/lr_{sa_lr}_{ca_lr}_epochs_{epochs}/{epochs}'
        # dir = '~/extract-concept/Lora_model/simple_concept_Picasso_negative/lr_0.001_epochs_20/20'
        unet.load_attn_procs(dir)
        
        for key in diffuser.unet.attn_processors.keys():
            diffuser.unet.attn_processors[key].to_q_lora.up.weight.data *= -1.0
            diffuser.unet.attn_processors[key].to_k_lora.up.weight.data *= -1.0
            diffuser.unet.attn_processors[key].to_v_lora.up.weight.data *= -1.0
            diffuser.unet.attn_processors[key].to_out_lora.up.weight.data *= -1.0
        
        # scale = 1.0
        # for key in unet.attn_processors.keys():
        #     unet_layer = extract_key(key)
        #     for i in ["q","k","v"]:
        #         exec(f"unet.{unet_layer}.to_{i}.weight.data -= {scale}*(torch.matmul(unet.attn_processors['{key}'].to_{i}_lora.up.weight.data,unet.attn_processors['{key}'].to_{i}_lora.down.weight.data))" )
        #     exec(f"unet.{unet_layer}.to_out[0].weight.data -= {scale}*(torch.matmul(unet.attn_processors['{key}'].to_out_lora.up.weight.data,unet.attn_processors['{key}'].to_out_lora.down.weight.data))" )
        
        # lora_attn_procs = {}
        # for name in unet.attn_processors.keys():
        #     cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        #     if name.startswith("mid_block"):
        #         hidden_size = unet.config.block_out_channels[-1]
        #     elif name.startswith("up_blocks"):
        #         block_id = int(name[len("up_blocks.")])
        #         hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        #     elif name.startswith("down_blocks"):
        #         block_id = int(name[len("down_blocks.")])
        #         hidden_size = unet.config.block_out_channels[block_id]

        #     lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,rank = rank)

        # # set lora
        # unet.set_attn_processor(lora_attn_procs)
            
    sa_layers= dict()
    ca_layers = dict()
    for i in unet.attn_processors.keys():
        if "attn1" in i:
            sa_layers[i] = unet.attn_processors[i]
        else:
            ca_layers[i] = unet.attn_processors[i]
    sa_layers = AttnProcsLayers(sa_layers)
    ca_layers = AttnProcsLayers(ca_layers)

    diffuser = diffuser.to('cuda:0')
    optimizer = torch.optim.Adam([
        {'params':sa_layers.parameters(),'lr':sa_lr,
            },
        {'params':ca_layers.parameters(),'lr':ca_lr,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    """
    prepare dataset
    """
    # train_data = ImgCaptionSet(root_dir=data_path)
    # train_loader = torch.utils.data.DataLoader(
    #     train_data,
    #     shuffle = True,
    #     batch_size = batchsize,
    # )

    """
    training
    """
    weight_dtype = torch.float32
    for epoch in tqdm(range(epochs)):
        os.environ["IMAGERY"] = "Sunflowers"
        create_data(freezed,seed=epoch,data_path=data_path,type=type)
        train_data = ImgCaptionSet(root_dir=data_path)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            shuffle = True,
            batch_size = batchsize,
        )

        # scale = random.uniform(0,2)
        # if scale>1:
        #     scale = sqrt(1/1.1)
        # else:
        #     scale = sqrt(1.1)
        # for key in diffuser.unet.attn_processors.keys():
        #     diffuser.unet.attn_processors[key].to_q_lora.up.weight.data *= scale
        #     diffuser.unet.attn_processors[key].to_k_lora.up.weight.data *= scale
        #     diffuser.unet.attn_processors[key].to_v_lora.up.weight.data *= scale
        #     diffuser.unet.attn_processors[key].to_out_lora.up.weight.data *= scale
            
        #     diffuser.unet.attn_processors[key].to_q_lora.down.weight.data *= scale
        #     diffuser.unet.attn_processors[key].to_k_lora.down.weight.data *= scale
        #     diffuser.unet.attn_processors[key].to_v_lora.down.weight.data *= scale
        #     diffuser.unet.attn_processors[key].to_out_lora.down.weight.data *= scale 
        
        for iteration in range(iterations):
            unet.train()
            
            train_loss = 0.0
            for step,batch in enumerate(train_loader):
                img,caption = batch
                # convert caption to embedding
                text_tokens = diffuser.tokenizer(caption, padding="max_length", max_length=diffuser.tokenizer.model_max_length, truncation=True, return_tensors="pt").to("cuda:0")

                text_embeddings = diffuser.text_encoder(**text_tokens)[0]
                text_embeddings = text_embeddings.to('cuda:0')
                # convert image to latent space
                img = img.to('cuda:0')
                latents = diffuser.vae.encode(img.to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * diffuser.vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(1,nsteps-1,(bsz,),device=latents.device)
                noisy_latents = diffuser.scheduler.add_noise(latents, noise, timesteps)

                target = noise
                pred = unet(noisy_latents, timesteps, text_embeddings).sample
                # input(pred)
                loss = F.mse_loss(pred.float(),target.float(), reduction="mean")

                
                loss.backward()
                train_loss+=loss
                params_to_clip = ca_layers.parameters()
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
                params_to_clip = sa_layers.parameters()
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
            print(train_loss)
            # scheduler.step()

        output_dir = os.path.join(output_root_dir,f"{epoch+1}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # print(output_dir)
        unet.save_attn_procs(output_dir)


if __name__ =='__main__':
    
    train_lora(type="negative")
    train_lora(type="positive")   

    # train_data = ImgCaptionSet(root_dir=data_path)
    # for i,c in train_data:
    #     print(i.shape)
