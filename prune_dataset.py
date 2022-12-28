# filter.py
import torch
from random import random, randint, choice, shuffle
from vit_pytorch import ViT
import numpy as np
import os
import json

from cross_efficient_vit import CrossEfficientViT
import uuid
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
import cv2
from transforms_cevit.albu import IsotropicResize
import glob
import pandas as pd
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight 
from torch.optim import lr_scheduler
import collections
import math
import yaml
import argparse
from pathlib import Path
import textwrap 


experiment_name = "E-Vit-Exp10"
checkpoint_path = "E-Vit-Exp10_efficientnet_step730000_All"
model_path = "Checkpoints" + "/" + experiment_name + "/" + checkpoint_path

config_file = "configs/architecture.yaml"


dataset_path_labels_fake = ["/CelebDF/", "/deeper_forensics/", "/deepfacelab/", "/DFDC/", "/Face_Forensics/", "/Face2Face/", "/FaceShifter/", "/FaceSwap/", "/FF_GoogleDF/", "/NeuralTextures/"]
dataset_path_labels_real = ["/CelebDF/", "/deeper_forensics/", "/DFDC/", "/Face_Forensics/", "/VoxCeleb/"]
 

with open(config_file, 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

model = CrossEfficientViT(config=config)
model.eval()   
model.load_state_dict(torch.load(model_path))
model.cuda()
train_dataset = DeepFakesDataset(dataset_path_labels_fake, dataset_path_labels_real, config['model']['image-size'])
    
dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                pin_memory=False, drop_last=True, timeout=0,
                                worker_init_fn=None, prefetch_factor=2,
                                persistent_workers=False)
del train_dataset

validation_dataset = DeepFakesDataset(dataset_path_labels_fake, dataset_path_labels_real, config['model']['image-size'], mode='validation')
val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                pin_memory=False, drop_last=True, timeout=0,
                                worker_init_fn=None, prefetch_factor=2,
                                persistent_workers=False)
del validation_dataset
# output_pred_txt_path = fake_image_path.replace("dmdf_v2/", "dmdf_v2_evaluation_results/" + experiment_name + "/" + checkpoint_path + "/" ).replace(".png", ".txt")

# Loop through entire training dataset
for index, (images, labels, image_path_names) in enumerate(dl):
    global_step += 1
    images = np.transpose(images, (0, 3, 1, 2))
    labels = labels.unsqueeze(1)
    images = images.cuda()
    
    y_pred = model(images)
    y_pred = y_pred.cpu()


# for index, (val_images, val_labels, val_image_path_names) in enumerate(val_dl):

#     val_images = np.transpose(val_images, (0, 3, 1, 2))

#     val_images = val_images.cuda()
#     val_labels = val_labels.unsqueeze(1)
#     val_pred = model(val_images)
#     val_pred = val_pred.cpu()