
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '9'
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import copy
import diffusers
from diffusers import StableDiffusionPipeline,EulerDiscreteScheduler,DDIMScheduler,EulerAncestralDiscreteScheduler,DDPMScheduler
# from StableDiffuser import StableDiffuser
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers
from tqdm import tqdm
import random
import numpy as np
from math import sqrt
import sys
import re
data = '/home/czhou/Cplug-in/data/'
def extract_key(key):
    key = key.replace('.processor','')
    dot_digitals = re.findall('\.\d',key)
    for dot_digital in dot_digitals:
        digital = re.search('\d',dot_digital).group()
        target = f'[{digital}]'
        key = key.replace(dot_digital,target)
    return key

output_root_dir = data+"merged_lora"
raw_styles = [
    "Leonardo da Vinci",
    "Vincent van Gogh",
    # "Pablo Picasso",
    # "Claude Monet",
    # "Michelangelo",
    # "Rembrandt van Rijn",
    # "Salvador Dalí",
    # "Johannes Vermeer",
    # "Frida Kahlo",
    # "Henri Matisse" ,
]

styles = [style.replace(" ", "_") for style in raw_styles]
loras = [f"/home/czhou/Cplug-in/data/Lora_model_for_merge/{style}_positive/lr_5e-05_0.0001_epochs_10/10" for style in styles]



sa_lr = 2e-2
ca_lr = 2e-2

batchsize = 4
sample_images=1

imagery = "simple contents"
# style = "Picasso"
# imagery = sys.argv[1]
# style = sys.argv[2]

resolution = 512
epochs = 100
iterations = 10
nsteps = 20
max_grad_norm = 1.0
rank = 140


train_transforms = transforms.Compose(
        [   transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
         
            transforms.Resize((resolution,resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5,], [0.5]),
        ]
    )

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
            caption = f'the painitng of {imagery}, high definition, high quality, diverse'
        return img,caption

class LayerOutputHook:  
    def __init__(self):  
        self.layer_outputs = []  
        self.layer_inputs = []
        self.layers = []
    def __call__(self, module, inputs, output):  
        self.layer_outputs.append(output)  
        self.layer_inputs.append(inputs[0])
        self.layers.append(module)
    def clear(self):  
        self.layer_inputs = []
        self.layer_outputs = []  
        self.layers = [] 

def seed_torch(seed=1029):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(0)




from diffusers import AutoencoderKL, UNet2DConditionModel,StableDiffusionPipeline,EulerAncestralDiscreteScheduler,DDPMScheduler
model_id = "runwayml/stable-diffusion-v1-5"
model_id = "/home/czhou/data/sd-1.5/"
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
diffuser = StableDiffusionPipeline.from_pretrained(model_id,
                                                   scheduler = scheduler,
                                                    safety_checker = None,
                                                    # requires_safety_checker = False
                                                   ).to("cuda")
freezed = copy.deepcopy(diffuser)

unet = diffuser.unet

# lora setup
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
unet.set_attn_processor(lora_attn_procs)

# set hook
lora_layers = AttnProcsLayers(unet.attn_processors)
hook = LayerOutputHook()  
for _,layer in lora_layers.named_modules():
    if type(layer) == diffusers.models.attention_processor.LoRALinearLayer:
        layer.register_forward_hook(hook)

# optimizer setup
sa_layers= dict()
ca_layers = dict()
for i in unet.attn_processors.keys():
    if "attn1" in i:
        sa_layers[i] = unet.attn_processors[i]
    else:
        ca_layers[i] = unet.attn_processors[i]
sa_layers = AttnProcsLayers(sa_layers)
ca_layers = AttnProcsLayers(ca_layers)
sa_layers.requires_grad_(True)
ca_layers.requires_grad_(True)

diffuser = diffuser.to('cuda')
unet.requires_grad_(True)
optimizer = torch.optim.Adam(
    [
    {'params':sa_layers.parameters(),'lr':sa_lr,
        },
    {'params':ca_layers.parameters(),'lr':ca_lr,
        },
    ]
    # [{'params':unet.parameters(),'lr':1},]
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)



        
for epoch in range(epochs):
    unet.train()

    generator = torch.manual_seed(0)
    hook.clear()
    
    train_loss = 0.0
    generator = torch.manual_seed(epoch)
    for step,batch in enumerate(range(1)):
        # img,caption = batch
        img = torch.randint(1,(1,3,512,512))
        caption = f'the painitng of {imagery}, high definition, high quality, diverse'
        weight_dtype = torch.float

        text_tokens = diffuser.tokenizer(caption, padding="max_length", max_length=diffuser.tokenizer.model_max_length, truncation=True, return_tensors="pt").to("cuda")

        text_embeddings = diffuser.text_encoder(**text_tokens)[0]
        text_embeddings = text_embeddings.to('cuda')
        # convert image to latent space
        img = img.to('cuda')
        latents = diffuser.vae.encode(img.to(dtype=weight_dtype)).latent_dist.sample()
        latents = latents * diffuser.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        target = noise

        timesteps = torch.randint(1,nsteps-1,(bsz,),device=latents.device)
        noisy_latents = diffuser.scheduler.add_noise(latents, noise, timesteps)

        loss = None
        pred1 = diffuser.unet(noisy_latents, timesteps, text_embeddings).sample

        for lora in loras:

            text_tokens = diffuser.tokenizer(caption, padding="max_length", max_length=diffuser.tokenizer.model_max_length, truncation=True, return_tensors="pt").to("cuda")
            text_embeddings = diffuser.text_encoder(**text_tokens)[0]
            text_embeddings = text_embeddings.to('cuda')
            freezed.unet.load_attn_procs(lora)
            lora_layers = AttnProcsLayers(freezed.unet.attn_processors)
            freezed_hook = LayerOutputHook()  
            for _,layer in lora_layers.named_modules():
                if type(layer) == diffusers.models.attention_processor.LoRALinearLayer:
                    layer.register_forward_hook(freezed_hook)
            # scale = 1.0
            # for key in freezed.unet.attn_processors.keys():
            #     # if "attn2" in key:
            #     #     continue
            #     # print(key)
            #     unet_layer = extract_key(key)
            #     for i in ["q","k","v","out"]:
            #         w = dict()                    
            #         exec(f"w['0']={scale}*(torch.matmul(freezed.unet.attn_processors['{key}'].to_{i}_lora.up.weight.data,freezed.unet.attn_processors['{key}'].to_{i}_lora.down.weight.data))")
            #         if i == "out":
            #             exec(f"freezed.unet.{unet_layer}.to_out[0].weight.data += w['0']")
            #         else:        
            #             exec(f"freezed.unet.{unet_layer}.to_{i}.weight.data += w['0']" )

            # freezed.unet.set_default_attn_processor()

            
            pred2 = freezed.unet(noisy_latents, timesteps, text_embeddings).sample
            # # end to end optimize
            # # input(len(freezed_hook.layer_outputs))
            # if loss == None:
            #     loss = sum([F.mse_loss(i,i1,reduction="mean") for i,i1 in zip(hook.layer_outputs,freezed_hook.layer_outputs)])/len(hook.layer_outputs)
            # else:
            #     loss += sum([F.mse_loss(i,i1,reduction="mean") for i,i1 in zip(hook.layer_outputs,freezed_hook.layer_outputs)])/len(hook.layer_outputs)
            
            # # layerwise optimize
            outputs = []
            for i,inputs in enumerate(freezed_hook.layer_inputs):
                outputs.append(hook.layers[i](inputs))   
                # input(outputs)      
                
            layer1_loss = F.mse_loss(outputs[0],freezed_hook.layer_outputs[0],reduction="sum")
            print(f"{epoch},layer1_loss = {layer1_loss}")      
            layer128_loss = F.mse_loss(outputs[-4],freezed_hook.layer_outputs[-4],reduction="sum")
            print(f"{epoch},layer-1_loss = {layer128_loss}")
            if loss == None:
                loss = sum([F.mse_loss(i,i1,reduction="mean") for i,i1 in zip(outputs,freezed_hook.layer_outputs)])/len(outputs)
            else:
                loss += sum([F.mse_loss(i,i1,reduction="mean") for i,i1 in zip(outputs,freezed_hook.layer_outputs)])/len(outputs)
 
            freezed_hook.clear()
            # print(loss)
            print(epoch,loss)
            # print(loss.grad)
            # input(optimizer.param_groups[0]['params'][0])

            loss.backward()
            hook.clear()
            # print(unet)
            params_to_clip = ca_layers.parameters()
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
            params_to_clip = sa_layers.parameters()
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            del loss
            torch.cuda.empty_cache()
        
    output_dir = os.path.join(output_root_dir,f"{epoch+1}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # print(output_dir)
    unet.save_attn_procs(output_dir)