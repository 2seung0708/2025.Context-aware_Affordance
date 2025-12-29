from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os

import numpy as np
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import cv2
def _convert_image_to_rgb(image):
    return image.convert("RGB")

import json
import clip

data_dir= "../../../AGK20K"

gt_dir = "../../../06_Affordance-R1/Affordance-R1/vis_results_refine_anno_v2/sampled/AGD20K_refine/"

input_paths = []
gt_paths = []
input_verbs = []
input_nouns = []
verb2vid = {}
vid2verb = []
noun2nid = {}
nid2noun = []
## 추가됨 ----------------------
input_active=[]
input_passive=[]
active2aid ={}
aid2active = []
passive2pid ={}
pid2passive = []


input_parts=[]
parts2partid ={}
partid2parts = []
## ----------------------------

noun2verbs = {}
noun2irrelevant_nouns = {}


## trainset
for iv, verb in enumerate(sorted(os.listdir(os.path.join(gt_dir, "Seen", "trainset", "egocentric")))):
    verb2vid[verb] = iv
    vid2verb.append(verb)
    for noun in sorted(os.listdir(os.path.join(gt_dir, "Seen", "trainset", "egocentric", verb))):       
        if noun not in noun2nid:
            noun2nid[noun] = len(noun2nid)
            nid2noun.append(noun)
for iv, verb in enumerate(os.listdir(os.path.join(gt_dir, "Seen", "trainset", "egocentric"))):
    for noun in os.listdir(os.path.join(gt_dir, "Seen", "trainset", "egocentric", verb)):  
        if noun not in noun2verbs:
            noun2verbs[noun] = []
        noun2verbs[noun].append(verb)
        for f in os.listdir(os.path.join(gt_dir, "Seen", "trainset", "egocentric", verb, noun)):
            if ".jpg" in f:      
                input_paths.append(os.path.join(data_dir, "Seen", "trainset", "egocentric", verb, noun, f))
                input_verbs.append(verb)
                input_nouns.append(noun)
                
            ## json 데이터 추가
            if ".json" in f:  
                with open(os.path.join(gt_dir, "Seen", "trainset", "egocentric", verb, noun, f), encoding='utf-8') as file_: j_data = json.load(file_)
                try:
                    input_active.append(j_data[0]['active'])
                    input_passive.append(j_data[0]['passive'])
                    input_parts.append(j_data[0]["part name"])
                except:
                    # print(j_data)
                    print(os.path.join(gt_dir, "Seen", "trainset", "egocentric", verb, noun, f))
                    continue
                    # import pdb;pdb.set_trace()
                if  j_data[0]['active'] not in active2aid:
                    active2aid[j_data[0]['active']] = len(active2aid)
                    aid2active.append(j_data[0]['active'])

                if  j_data[0]['passive'] not in passive2pid:
                    passive2pid[j_data[0]['passive']] = len(passive2pid)
                    pid2passive.append(j_data[0]['passive'])

                if j_data[0]["part name"] not in parts2partid:
                    
                    parts2partid[j_data[0]["part name"]] = len(parts2partid)
                    partid2parts.append(j_data[0]["part name"])


## testset
## trainset
for iv, verb in enumerate(sorted(os.listdir(os.path.join(gt_dir, "Seen", "testset", "egocentric")))):
    verb2vid[verb] = iv
    vid2verb.append(verb)
    for noun in sorted(os.listdir(os.path.join(gt_dir, "Seen", "testset", "egocentric", verb))):       
        if noun not in noun2nid:
            noun2nid[noun] = len(noun2nid)
            nid2noun.append(noun)
for iv, verb in enumerate(os.listdir(os.path.join(gt_dir, "Seen", "testset", "egocentric"))):
    for noun in os.listdir(os.path.join(gt_dir, "Seen", "testset", "egocentric", verb)):  
        if noun not in noun2verbs:
            noun2verbs[noun] = []
        noun2verbs[noun].append(verb)
        for f in os.listdir(os.path.join(gt_dir, "Seen", "testset", "egocentric", verb, noun)):
            if ".jpg" in f:      
                input_paths.append(os.path.join(data_dir, "Seen", "testset", "egocentric", verb, noun, f))
                input_verbs.append(verb)
                input_nouns.append(noun)
                
            ## json 데이터 추가
            if ".json" in f:  
                with open(os.path.join(gt_dir, "Seen", "testset", "egocentric", verb, noun, f), encoding='utf-8') as file_: j_data = json.load(file_)
                try:
                    input_active.append(j_data[0]['active'])
                    input_passive.append(j_data[0]['passive'])
                    input_parts.append(j_data[0]["part name"])
                except:
                    # print(j_data)
                    print(os.path.join(gt_dir, "Seen", "testset", "egocentric", verb, noun, f))
                    continue
                    # import pdb;pdb.set_trace()

                if  j_data[0]['active'] not in active2aid:
                    active2aid[j_data[0]['active']] = len(active2aid)
                    aid2active.append(j_data[0]['active'])

                if  j_data[0]['passive'] not in passive2pid:
                    passive2pid[j_data[0]['passive']] = len(passive2pid)
                    pid2passive.append(j_data[0]['passive'])


                if j_data[0]["part name"] not in parts2partid:
                    
                    parts2partid[j_data[0]["part name"]] = len(parts2partid)
                    partid2parts.append(j_data[0]["part name"])


## 처음 한번만.. 그 다음에는 load 해와서 쓰기 --------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)

with torch.no_grad():
    tokens = clip.tokenize(aid2active).to(device)                 # ["knife", "stem of apple", ...]
    feats  = model.encode_text(tokens)                                  # [N, D]
    feats  = feats / feats.norm(dim=-1, keepdim=True)                   # L2 정규화

activeFeat = {k: v.detach().cpu() for k, v in zip(aid2active, feats)} 
# import pdb;pdb.set_trace()
torch.save(activeFeat, os.path.join(gt_dir, "custom_sentenceFeatActive.pth"))

with torch.no_grad():
    tokens = clip.tokenize(pid2passive).to(device)                 # ["knife", "stem of apple", ...]
    feats  = model.encode_text(tokens)                                  # [N, D]
    feats  = feats / feats.norm(dim=-1, keepdim=True)                   # L2 정규화

apassiveFeat = {k: v.detach().cpu() for k, v in zip(pid2passive, feats)} 
torch.save(apassiveFeat, os.path.join(gt_dir, "custom_sentenceFeatPassive.pth"))


import pdb;pdb.set_trace()
with torch.no_grad():
    tokens = clip.tokenize(partid2parts).to(device)                 # ["knife", "stem of apple", ...]
    feats  = model.encode_text(tokens)                                  # [N, D]
    feats  = feats / feats.norm(dim=-1, keepdim=True)                   # L2 정규화

partFeat = {k: v.detach().cpu() for k, v in zip(partid2parts, feats)} 
torch.save(partFeat, os.path.join(gt_dir, "custom_sentenceFeatPart.pth"))

# nounsFeat = torch.load(os.path.join(data_dir, "sentenceFeatNounAGD.pth"))
# verbsFeat = torch.load(os.path.join(data_dir, "sentenceFeatVerbAGD.pth"))
# partsFeat = torch.load(os.path.join(data_dir, "sentenceFeatPartAGD.pth"))