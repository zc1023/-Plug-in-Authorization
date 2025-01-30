# Dino distance
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np

def dino_distance(image1, image2):
    processor = AutoImageProcessor.from_pretrained('/home/czhou/data/dinov2-base')
    model = AutoModel.from_pretrained('/home/czhou/data/dinov2-base')
    image1 = Image.open(image1).convert('RGB')
    image2 = Image.open(image2).convert('RGB')
    inputs = processor(images=[image1,image2], return_tensors="pt")
    outputs = model(**inputs)
    # print(outputs)
    pooler_output = outputs.pooler_output
    img1_embedding = pooler_output[0].detach().numpy()
    img2_embedding = pooler_output[1].detach().numpy()
    norm1 =  np.linalg.norm(img1_embedding)
    norm2 =  np.linalg.norm(img2_embedding)
    dot_product = np.dot(img1_embedding, img2_embedding)
    distance = abs(dot_product / (norm1 * norm2))
    return distance

# clip distance
import torch

from PIL import Image
import os

import pandas as pd

from sentence_transformers import SentenceTransformer, util
from PIL import Image
import shutil

def get_clip_score(images, text, w=1):
    images = [Image.open(image) for image in images]    

    #Load CLIP model
    model = SentenceTransformer("/home/czhou/data/clip-ViT-B-32")

    #Encode an image:
    img_emb = model.encode(images)

    #Encode text descriptions
    text_emb = model.encode(text)

    #Compute cosine similarities 
    cos_scores = util.cos_sim(img_emb, text_emb)
    return cos_scores

# fid 
def fid_score(orign,non_infring):
    score = fid.compute_fid(orign,non_infring, mode="clean",device = torch.device("cuda"))
    return score

def kid_score(orign,non_infring):
    score = fid.compute_kid(orign,non_infring, mode="clean",device = torch.device("cuda"))
    return score


import torch
from torch_fidelity import calculate_metrics
def fid_kid_score(orign,non_infring):
    metrics = calculate_metrics(
        input1 = orign, 
        input2 = non_infring,
        cuda=True,
        isc=False,
        kid=True,
        fid=True,
        kid_subset_size=2,
        verbose=False
    )
    return metrics["frechet_inception_distance"],metrics["kernel_inception_distance_mean"]


# LPIPS

import torch
from PIL import Image
from torchvision import transforms
import lpips

def LPIPS_score(img_path1, img_path2):
    # 初始化LPIPS模型
    loss_fn_alex = lpips.LPIPS(net='alex')  # 使用AlexNet作为特征提取器
    # 或者使用vgg:
    # loss_fn_vgg = lpips.LPIPS(net='vgg')

    # 加载并预处理图像
    def load_image(image_path, transform=None):
        img = Image.open(image_path).convert('RGB')
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),  # 调整大小到256x256
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
            ])
        return transform(img).unsqueeze(0)  # 添加batch维度


    # 加载图像
    img_tensor1 = load_image(img_path1)
    img_tensor2 = load_image(img_path2)

    # 确保两个张量在相同的设备上 (CPU或GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_alex = loss_fn_alex.to(device)
    img_tensor1 = img_tensor1.to(device)
    img_tensor2 = img_tensor2.to(device)

    # 计算LPIPS距离
    distance = loss_fn_alex(img_tensor1, img_tensor2)

    return distance.detach().to('cpu')

styles = [
    #  "Leonardo da Vinci", 
 "Vincent van Gogh" ,
#  "Pablo Picasso" ,
#  "Claude Monet" ,
#  "Michelangelo" ,
#  "Rembrandt van Rijn", 
#  "Salvador Dalí" ,
#  "Johannes Vermeer", 
#  "Frida Kahlo" ,
#  "Henri Matisse",
]

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
styles = [style.replace(" ", "_") for style in styles]

# # seen contents
target_style_dir = "/home/czhou/Cplug-in/data/target_style"
fids = []
kids = []
lpipss = []
dino = []
clips = []
for style in styles:
    style_dir = os.path.join(target_style_dir, style)
    # for i in range(20):
    #     false_dir =  os.path.join(style_dir, str(i),"False")
    #     true_dir  =  os.path.join(style_dir, str(i),"True")
    #     fid,kid = fid_kid_score(false_dir, true_dir)
    #     fids.append(fid)
    #     kids.append(kid)
    for i in range(20):
        false_dir =  os.path.join(style_dir, str(i),"False")
        true_dir  =  os.path.join(style_dir, str(i),"True")
        for j in range(2):
            false_image = os.path.join(false_dir, str(j)+"_False.jpg")
            true_image  = os.path.join(true_dir, str(j)+"_True.jpg")
            lpipss.append(LPIPS_score(false_image, true_image)[0][0][0][0])
            dino.append(dino_distance(false_image, true_image))
            clips.append(get_clip_score([true_image],f"{imageries[i]} by Vincent van Gogh")[0][0])

with open("./target_result.txt","w") as f:
    f.write("lpips\tdino\tclip\n")
    for i in range(len(lpipss)):
        f.write( str(lpipss[i]) + "\t" + str(dino[i]) + "\t" + str(clips[i])+"\n")

# with open("./target_result_fid.txt","w") as f:
#     f.write("fid\tkid\n")
#     for i in range(len(fids)):
#         f.write(str(fids[i]) + "\t" + str(kids[i]) + "\n")


# target_style_dir = "/home/czhou/Cplug-in/data/surrounding_style"
# fids = []
# kids = []
# lpipss = []
# dino = []
# clips = []
# for style in styles:
#     style_dir = os.path.join(target_style_dir, style)
#     for i in range(20):
#         false_dir =  os.path.join(style_dir, str(i),"False")
#         true_dir  =  os.path.join(style_dir, str(i),"True")
#         fid,kid = fid_kid_score(false_dir, true_dir)
#         fids.append(fid)
#         kids.append(kid)
#     for i in range(20):
#         j = 1
#         false_dir =  os.path.join(style_dir, str(i),"False")
#         true_dir  =  os.path.join(style_dir, str(i),"True")
#         false_image = os.path.join(false_dir, str(j)+"_False.jpg")
#         true_image  = os.path.join(true_dir, str(j)+"_True.jpg")
#         lpipss.append(LPIPS_score(false_image, true_image)[0][0][0][0])
#         dino.append(dino_distance(false_image, true_image))
#         clips.append(get_clip_score([true_image],f"{imageries[i]} by Kazimir Malevich")[0][0])

# with open("./surrounding_result.txt","w") as f:
#     f.write("lpips\tdino\tclip\n")
#     for i in range(len(lpipss)):
#         f.write( str(lpipss[i]) + "\t" + str(dino[i]) + "\t" + str(clips[i])+"\n")

# with open("./surrounding_result_fid.txt","w") as f:
#     f.write("fid\tkid\n")
#     for i in range(len(fids)):
#         f.write(str(fids[i]) + "\t" + str(kids[i]) + "\n")