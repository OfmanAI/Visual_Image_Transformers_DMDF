import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, choice, shuffle
from vit_pytorch import ViT
import numpy as np
import os
import json
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from cross_efficient_vit import CrossEfficientViT
import uuid
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
import cv2
from transforms_cevit.albu import IsotropicResize
import glob
import pandas as pd
from tqdm import tqdm
from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params, check_correct_with_paths, get_dataset_from_imagepath, get_demographic_info_from_imagepath
from sklearn.utils.class_weight import compute_class_weight 
from torch.optim import lr_scheduler
import collections
from deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import textwrap 

# Main body
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=16, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='All', 
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|CelebDF|DeepFaceLab|All)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', default="configs/architecture.yaml", type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--patience', type=int, default=5, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    
    opt = parser.parse_args()
    experiment_name = "E-Vit-Exp9"
    print(opt)

    writer = SummaryWriter("logs2/" + experiment_name)
    MODELS_PATH = "Checkpoints/" + experiment_name + "/"

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
 
    model = CrossEfficientViT(config=config)
    model.train()   
    
    dataset_path_labels_fake = ["/CelebDF/", "/deeper_forensics/", "/deepfacelab/", "/DFDC/", "/Face_Forensics/", "/Face2Face/", "/FaceShifter/", "/FaceSwap/", "/NeuralTextures/"]
    dataset_path_labels_real = ["/CelebDF/", "/deeper_forensics/", "/DFDC/", "/Face_Forensics/", "/VoxCeleb/"]
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1
    else:
        print("No checkpoint loaded.")

    print("Model Parameters:", get_n_params(model))

    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_dataset = DeepFakesDataset(dataset_path_labels_fake, dataset_path_labels_real, config['model']['image-size'])
    
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=True, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    del train_dataset

    validation_dataset = DeepFakesDataset(dataset_path_labels_fake, dataset_path_labels_real, config['model']['image-size'], mode='validation', max_images=100000)
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=True, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    del validation_dataset
    

    model = model.cuda()

    # Create training variables
    counter = 0
    global_step = 0;
    not_improved_loss = 0
    previous_loss = math.inf
    log_tb_training_steps = 100
    evaluation_steps = 1000
    save_checkpoint_steps = 10000
    num_log_images = 8

    # Run training loop
    for t in range(starting_epoch, opt.num_epochs + 1):
        if not_improved_loss == opt.patience:
            break
        counter = 0
        

        total_loss = 0
        total_val_loss = 0
        
        bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*config['training']['bs'])+len(val_dl))
        train_correct = 0
        positive = 0
        negative = 0

        total_correct_by_dataset_train = {}
        
        # Loop through entire training dataset
        for index, (images, labels, image_path_names) in enumerate(dl):
            model.train()  
            global_step += 1
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.unsqueeze(1)
            images = images.cuda()
            
            y_pred = model(images)
            y_pred = y_pred.cpu()
            loss = loss_fn(y_pred, labels)
        
            # Check positive and negative accuracy per dataset type
            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            correct_by_dataset = check_correct_with_paths(y_pred, labels, image_path_names) 

            # Combine results in each batch, will be averaged during logging
            if not total_correct_by_dataset_train:
                total_correct_by_dataset_train = correct_by_dataset
            else:
                for key, val in correct_by_dataset.items():
                    if(key not in total_correct_by_dataset_train):
                        total_correct_by_dataset_train[key] = {}
                    for key2, val2 in val.items():
                        if(key2 not in total_correct_by_dataset_train[key]):
                            total_correct_by_dataset_train[key][key2] = 0
                        total_correct_by_dataset_train[key][key2] += val2

            train_correct += corrects
            positive += positive_class
            negative += negative_class

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            counter += 1
            learning_rate = optimizer.param_groups[0]['lr']

            # Create an easy to read variable for logging
            total_loss += round(loss.item(), 2)

            # Log training losses
            if(global_step % log_tb_training_steps == 0):
                writer.add_scalar('Train/Loss: ', loss.item(), global_step)
                for correct_by_dataset_key, correct_by_dataset_value in total_correct_by_dataset_train.items():
                    TB_var_name_pos = "Train/Accuracy Positive - " + correct_by_dataset_key
                    TB_var_name_neg = "Train/Accuracy Negative - " + correct_by_dataset_key
                    if((correct_by_dataset_value["total_positive"] > 0)):
                        positive_accuracy = correct_by_dataset_value["correct_positive"]/(correct_by_dataset_value["total_positive"])
                        writer.add_scalar(TB_var_name_pos, positive_accuracy, global_step)
                    if((correct_by_dataset_value["total_negative"] > 0)):
                        negative_accuracy = correct_by_dataset_value["correct_negative"]/(correct_by_dataset_value["total_negative"])
                        writer.add_scalar(TB_var_name_neg, negative_accuracy, global_step)
                
                writer.add_scalar('Train/LR', learning_rate, global_step)
                writer.add_scalar('Train/Accuracy', corrects/config['training']['bs'], global_step)

                # Log images to investigate augmentations, labels, datasets, demographics, etc
                for image_index in range(0, num_log_images):

                    log_image = (images[image_index].detach().cpu().numpy() * 255.).astype(np.uint8)
                    log_image = np.uint8(np.clip(log_image, 0, 255))
                    log_image = np.transpose(log_image, (1, 2, 0))
                    log_image = cv2.cvtColor(log_image, cv2.COLOR_RGB2BGR)
                    log_image = cv2.resize(log_image, (0,0), fx=2.0, fy=2.0) 
                    log_image = cv2.copyMakeBorder(log_image, 256, 0, 0, 0, cv2.BORDER_CONSTANT)

                    log_path = image_path_names[image_index]
                    log_path_wrapped_list = textwrap.wrap(log_path, width=40)
                    
                    log_path_dataset_type = get_dataset_from_imagepath(log_path)
                    log_label = labels[image_index].detach().cpu().numpy()[0]
                    log_prediction = torch.sigmoid(y_pred[image_index]).detach().cpu().numpy()[0]

                    log_label_prediction = "LABEL: " + str(log_label) + "   |   " + str(log_prediction)

                    final_age, final_gender, final_race, final_emotion = get_demographic_info_from_imagepath(log_path)

                    for log_path_index, log_path_line in enumerate(log_path_wrapped_list):
                        cv2.putText(log_image, log_path_line, (4,16 * log_path_index + 16), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                    
                    cv2.putText(log_image, log_path_dataset_type, (4, 16 * 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                    cv2.putText(log_image, log_label_prediction, (4, 16 * 7), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

                    if(final_age is not None):
                        cv2.putText(log_image, "PREDICTED AGE:           " + str(final_age), (4, 16 * 8), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                        cv2.putText(log_image, "PREDICTED GENDER:        " + final_gender, (4, 16 * 9), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                        cv2.putText(log_image, "PREDICTED RACE:          " + final_race, (4, 16 * 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                        cv2.putText(log_image, "PREDICTED AVG EMOTION:  " + final_emotion, (4, 16 * 11), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


                    writer.add_images(f"Train/Image_{image_index}", log_image, global_step, dataformats="HWC")


                # Reset logging dictionary so we can refill from scratch
                total_correct_by_dataset_train = {}

                # Log images along with dataset, image path, label, prediction, and demographic information


            # Update Console Bar
            for i in range(config['training']['bs']):
                bar.next()

            # Print values to console
            if index%100 == 0:
                print("\nLoss: ", total_loss/counter, "Accuracy: ",train_correct/(counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive)  

            # Run and log evaluation
            if global_step % evaluation_steps == 0:
                print("Evaluating data")
                val_counter = 0
                val_correct = 0
                val_positive = 0
                val_negative = 0
                model.eval()
                total_correct_by_dataset_eval = {}
                for index, (val_images, val_labels, val_image_path_names) in enumerate(val_dl):
            
                    val_images = np.transpose(val_images, (0, 3, 1, 2))

                    val_images = val_images.cuda()
                    val_labels = val_labels.unsqueeze(1)
                    val_pred = model(val_images)
                    val_pred = val_pred.cpu()
                    val_loss = loss_fn(val_pred, val_labels)
                    total_val_loss += round(val_loss.item(), 2)
                    val_corrects, val_positive_class, val_negative_class = check_correct(val_pred, val_labels)

                    correct_by_dataset = check_correct_with_paths(y_pred, labels, image_path_names)  
                    if not total_correct_by_dataset_eval:
                        total_correct_by_dataset_eval = correct_by_dataset
                    else:
                        for key, val in correct_by_dataset.items():
                            for key2, val2 in val.items():
                                total_correct_by_dataset_eval[key][key2] += val2

                    val_correct += val_corrects
                    val_positive += val_positive_class
                    val_negative += val_negative_class
                    val_counter += 1
                    bar.next()
                    
                
                

                total_val_loss /= val_counter
                val_correct /= val_counter
                if previous_loss <= total_val_loss:
                    print("Validation loss did not improved")
                    not_improved_loss += 1
                else:
                    not_improved_loss = 0
                
                previous_loss = total_val_loss

                writer.add_scalar('Eval/Loss: ', total_val_loss, global_step)
                for correct_by_dataset_key, correct_by_dataset_value in total_correct_by_dataset_eval.items():
                    TB_var_name_pos = "Eval/Accuracy Positive - " + correct_by_dataset_key
                    TB_var_name_neg = "Eval/Accuracy Negative - " + correct_by_dataset_key
                    if((correct_by_dataset_value["total_positive"] > 0)):
                        positive_accuracy = correct_by_dataset_value["correct_positive"]/(correct_by_dataset_value["total_positive"])
                        writer.add_scalar(TB_var_name_pos, positive_accuracy, global_step)
                    if((correct_by_dataset_value["total_negative"] > 0)):
                        negative_accuracy = correct_by_dataset_value["correct_negative"]/(correct_by_dataset_value["total_negative"])
                        writer.add_scalar(TB_var_name_neg, negative_accuracy, global_step)
                
                writer.add_scalar('Eval/LR', learning_rate, global_step)
                writer.add_scalar('Eval/Accuracy', val_correct/config['training']['bs'], global_step)

                # Log images to investigate augmentations, labels, datasets, demographics, etc
                for image_index in range(0, num_log_images):

                    log_image = (val_images[image_index].detach().cpu().numpy() * 255.).astype(np.uint8)
                    log_image = np.uint8(np.clip(log_image, 0, 255))
                    log_image = np.transpose(log_image, (1, 2, 0))
                    log_image = cv2.cvtColor(log_image, cv2.COLOR_RGB2BGR)
                    log_image = cv2.resize(log_image, (0,0), fx=2.0, fy=2.0) 
                    log_image = cv2.copyMakeBorder(log_image, 256, 0, 0, 0, cv2.BORDER_CONSTANT)

                    log_path = val_image_path_names[image_index]
                    log_path_wrapped_list = textwrap.wrap(log_path, width=40)
                    
                    log_path_dataset_type = get_dataset_from_imagepath(log_path)
                    log_label = val_labels[image_index].detach().cpu().numpy()[0]
                    log_prediction = torch.sigmoid(val_pred[image_index]).detach().cpu().numpy()[0]

                    log_label_prediction = "LABEL: " + str(log_label) + "   |   " + str(log_prediction)

                    final_age, final_gender, final_race, final_emotion = get_demographic_info_from_imagepath(log_path)

                    for log_path_index, log_path_line in enumerate(log_path_wrapped_list):
                        cv2.putText(log_image, log_path_line, (4,16 * log_path_index + 16), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                    
                    cv2.putText(log_image, log_path_dataset_type, (4, 16 * 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                    cv2.putText(log_image, log_label_prediction, (4, 16 * 7), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

                    if(final_age is not None):
                        cv2.putText(log_image, "PREDICTED AGE:           " + str(final_age), (4, 16 * 8), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                        cv2.putText(log_image, "PREDICTED GENDER:        " + final_gender, (4, 16 * 9), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                        cv2.putText(log_image, "PREDICTED RACE:          " + final_race, (4, 16 * 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                        cv2.putText(log_image, "PREDICTED AVG EMOTION:  " + final_emotion, (4, 16 * 11), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


                    writer.add_images(f"Eval/Image_{image_index}", log_image, global_step, dataformats="HWC")

        
            # Save checkpoint
            if(global_step % save_checkpoint_steps == 0):
                if not os.path.exists(MODELS_PATH):
                    os.makedirs(MODELS_PATH)
                torch.save(model.state_dict(), os.path.join(MODELS_PATH,  experiment_name + "_efficientnet_step" + str(global_step) + "_" + opt.dataset))
                
