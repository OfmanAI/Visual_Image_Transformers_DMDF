# eval.py
# Run evaluation on model with test dataset
# Built to mimick an inference script to help production

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
import datetime

from utils import get_demographic_info_from_imagepath

experiment_name = "E-Vit-Exp10"
checkpoint_path = "E-Vit-Exp10_efficientnet_step730000_All"
model_path = "Checkpoints" + "/" + experiment_name + "/" + checkpoint_path
dataset_path = "dmdf_v2"
crop_type = "bounding-box-tight-v2-original-fps/"
config_file = "configs/architecture.yaml"

dataset_path_val = dataset_path + "/" + "validation" + "/"
dataset_path_train = dataset_path + "/" + "train"
dataset_path_test = dataset_path + "/" + "test" + "/"

dataset_path_labels_fake = ["/CelebDF/", "/deeper_forensics/", "/deepfacelab/", "/DFDC/", "/Face_Forensics/", "/Face2Face/", "/FaceShifter/", "/FaceSwap/", "/FF_GoogleDF/", "/NeuralTextures/"]
dataset_path_labels_real = ["/CelebDF/", "/deeper_forensics/", "/DFDC/", "/Face_Forensics/", "/VoxCeleb/"]
dataset_path_labels_all = ["/deeper_forensics/", "/DFDC/", "/Face_Forensics/", "/Face2Face/", "/FaceShifter/", "/FaceSwap/", "/NeuralTextures/", "/deepfacelab/", "/CelebDF/", "/VoxCeleb/", "/FF_GoogleDF/"]

evaluation_image_paths_test_fake = []
evaluation_image_paths_test_real = []

for dataset_path_label in dataset_path_labels_fake:

    all_fake_images = []
    all_fake_images += sorted(Path(dataset_path_test + "/fake" + dataset_path_label + crop_type).glob('**/*.png'))
    all_fake_images += sorted(Path(dataset_path_val + "/fake" + dataset_path_label + crop_type).glob('**/*.png'))
    all_fake_images = list(set(all_fake_images))
    all_fake_images = [str(path) for path in all_fake_images]
    all_fake_images = sorted(all_fake_images)

    evaluation_image_paths_test_fake.extend(all_fake_images)

for dataset_path_label in dataset_path_labels_real:

    all_real_images = []
    all_real_images += sorted(Path(dataset_path_test + "/real" + dataset_path_label + crop_type).glob('**/*.png'))
    all_real_images += sorted(Path(dataset_path_val + "/real" + dataset_path_label + crop_type).glob('**/*.png'))
    all_real_images = list(set(all_real_images))
    all_real_images = [str(path) for path in all_real_images]
    all_real_images = sorted(all_real_images)

    evaluation_image_paths_test_real.extend(all_real_images)



with open(config_file, 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

model = CrossEfficientViT(config=config)
model.eval()   
model.load_state_dict(torch.load(model_path))
model.cuda()

print("TOTAL TEST FAKE IMAGES: " + str(len(evaluation_image_paths_test_fake)))
print("TOTAL TEST REAL IMAGES: " + str(len(evaluation_image_paths_test_real)))

evaluation_image_paths_test_all = []
# evaluation_image_paths_test_all.extend(evaluation_image_paths_test_fake)
# evaluation_image_paths_test_all.extend(evaluation_image_paths_test_real)
# Run through fake images

for image_path in evaluation_image_paths_test_all:

    output_pred_txt_path = image_path.replace("dmdf_v2/", "dmdf_v2_evaluation_results/" + experiment_name + "/" + checkpoint_path + "/" ).replace(".png", ".txt")

    if os.path.exists(output_pred_txt_path):
        continue

    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (config['model']['image-size'], config['model']['image-size']), interpolation = cv2.INTER_CUBIC)
    image = image/255.0
    image_tensor = torch.tensor([image]).float()

    images = np.transpose(image_tensor, (0, 3, 1, 2)).cuda()
    y_pred = model(images)
    y_pred = y_pred.cpu()
    preds = [np.asarray(torch.sigmoid(pred).detach().numpy()) for pred in y_pred]

    preds_final = preds[0][0]

    

    os.makedirs(os.path.dirname(output_pred_txt_path), exist_ok=True)
    with open(output_pred_txt_path, 'w') as f:
        f.write(str(preds_final))

    print(image_path, preds_final)

# Get all generated text files
output_txt_paths = "dmdf_v2_evaluation_results/" + experiment_name + "/" + checkpoint_path + "/" 
all_text_files = []
all_text_files += sorted(Path(output_txt_paths).glob('**/*.txt'))
all_text_files = list(set(all_text_files))
all_text_files = [str(path) for path in all_text_files]
all_text_files = sorted(all_text_files)
print("ALL TEXT FILES: " + str(len(all_text_files)))
# Combine all txt files into a single csv file
csv_list = [["Image Path", "Dataset", "Video Path", "Label", "Prediction", "Predicted Age", "Predicted Gender", "Predicted Race", "Predicted Emotion"]]
a = datetime.datetime.now()
for i, text_file in enumerate(all_text_files):

    original_image_path = text_file.replace(output_txt_paths, "dmdf_v2/").replace(".txt", ".png")
    prediction_value = None
    with open(text_file, 'r') as file:
        prediction_value = float(file.read().rstrip())

    final_age, final_gender, final_race, final_emotion = get_demographic_info_from_imagepath(original_image_path)

    this_item_dataset = None
    for dataset_path_label in dataset_path_labels_all:
        if(dataset_path_label in original_image_path):
            this_item_dataset = dataset_path_label

    this_item_label = None
    if("/fake/" in original_image_path):
        this_item_label = 1
    if("/real/" in original_image_path):
        this_item_label = 0

    this_item_video_path = os.path.dirname(original_image_path)
            
    this_item_list = [original_image_path, this_item_dataset, this_item_video_path, this_item_label, prediction_value, final_age, final_gender, final_race, final_emotion]
    if(i % 1000 == 0):
        print(i)
        print(datetime.datetime.now() - a)
        print(this_item_list)
        a = datetime.datetime.now()

    csv_list.append(this_item_list)

import csv

with open("out.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_list)
