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
from transforms.albu import IsotropicResize
import glob
import pandas as pd
from tqdm import tqdm
from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params, check_correct_with_paths
from sklearn.utils.class_weight import compute_class_weight 
from torch.optim import lr_scheduler
import collections
from deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter



# BASE_DIR = '../../deep_fakes/'
# DATA_DIR = os.path.join(BASE_DIR, "dataset")
# TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
# VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
# TEST_DIR = os.path.join(DATA_DIR, "test_set")
# MODELS_PATH = "models"
# METADATA_PATH = os.path.join(BASE_DIR, "data/metadata") # Folder containing all training metadata for DFDC dataset
# VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_val_labels.csv")


# def read_frames(video_path, train_dataset, validation_dataset):
    
#     # Get the video label based on dataset selected
#     method = get_method(video_path, DATA_DIR)
#     if TRAINING_DIR in video_path:
#         if "Original" in video_path:
#             label = 0.
#         elif "DFDC" in video_path:
#             for json_path in glob.glob(os.path.join(METADATA_PATH, "*.json")):
#                 with open(json_path, "r") as f:
#                     metadata = json.load(f)
#                 video_folder_name = os.path.basename(video_path)
#                 video_key = video_folder_name + ".mp4"
#                 if video_key in metadata.keys():
#                     item = metadata[video_key]
#                     label = item.get("label", None)
#                     if label == "FAKE":
#                         label = 1.         
#                     else:
#                         label = 0.
#                     break
#                 else:
#                     label = None
#         else:
#             label = 1.
#         if label == None:
#             print("NOT FOUND", video_path)
#     else:
#         if "Original" in video_path:
#             label = 0.
#         elif "DFDC" in video_path:
#             val_df = pd.DataFrame(pd.read_csv(VALIDATION_LABELS_PATH))
#             video_folder_name = os.path.basename(video_path)
#             video_key = video_folder_name + ".mp4"
#             label = val_df.loc[val_df['filename'] == video_key]['label'].values[0]
#         else:
#             label = 1.

#     # Calculate the interval to extract the frames
#     frames_number = len(os.listdir(video_path))
#     if label == 0:
#         min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-real']),1) # Compensate unbalancing
#     else:
#         min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-fake']),1)

    
    
#     if VALIDATION_DIR in video_path:
#         min_video_frames = int(max(min_video_frames/8, 2))
#     frames_interval = int(frames_number / min_video_frames)
#     frames_paths = os.listdir(video_path)
#     frames_paths_dict = {}

#     # Group the faces with the same index, reduce probabiity to skip some faces in the same video
#     for path in frames_paths:
#         for i in range(0,1):
#             if "_" + str(i) in path:
#                 if i not in frames_paths_dict.keys():
#                     frames_paths_dict[i] = [path]
#                 else:
#                     frames_paths_dict[i].append(path)
#     # Select only the frames at a certain interval
#     if frames_interval > 0:
#         for key in frames_paths_dict.keys():
#             if len(frames_paths_dict) > frames_interval:
#                 frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
            
#             frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]
#     # Select N frames from the collected ones
#     for key in frames_paths_dict.keys():
#         for index, frame_image in enumerate(frames_paths_dict[key]):
#             #image = transform(np.asarray(cv2.imread(os.path.join(video_path, frame_image))))
#             image = cv2.imread(os.path.join(video_path, frame_image))
#             if image is not None:
#                 if TRAINING_DIR in video_path:
#                     train_dataset.append((image, label))
#                 else:
#                     validation_dataset.append((image, label))

# Main body
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='All', 
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', default="configs/architecture.yaml", type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--patience', type=int, default=5, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    
    opt = parser.parse_args()
    print(opt)

    writer = SummaryWriter("logs2/CE-Vit-Exp2")

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
 
    model = CrossEfficientViT(config=config)
    model.train()   

    dataset_path_labels=["/deeper_forensics/", "/DFDC/", "/Face_Forensics/", "/Face2Face/", "/FaceShifter/", "/FaceSwap/", "/NeuralTextures/", "/deepfacelab/"]
    
    # all_fake_images_train = []
    # all_fake_images_train += sorted(Path("DMDF_Faces_V6/train/fake").glob('**/*.png'))
    # all_fake_images_train = list(set(all_fake_images_train))
    # all_fake_images_train = [str(path) for path in all_fake_images_train]
    # all_fake_images_train = sorted(all_fake_images_train)
    # shuffle(all_fake_images_train)

    # all_real_images_train = []
    # all_real_images_train += sorted(Path("DMDF_Faces_V6/train/real").glob('**/*.png'))
    # all_real_images_train = list(set(all_real_images_train))
    # all_real_images_train = [str(path) for path in all_real_images_train]
    # all_real_images_train = sorted(all_real_images_train)
    # shuffle(all_real_images_train)

    #len_fake = len(all_fake_images_train)
    #all_fake_images_val = all_fake_images_train[int(len_fake * 0.95):]
    #all_fake_images_train = all_fake_images_train[0:int(len_fake * 0.95)]

    #len_real = len(all_real_images_train)
    #all_real_images_val = all_real_images_train[int(len_real * 0.95):]
    #all_real_images_train = all_real_images_train[0:int(len_real * 0.95)]

    # all_fake_images_val = []
    # all_fake_images_val += sorted(Path("DMDF_Faces_V6/validation/fake").glob('**/*.png'))
    # all_fake_images_val = list(set(all_fake_images_val))
    # all_fake_images_val = [str(path) for path in all_fake_images_val]
    # all_fake_images_val = sorted(all_fake_images_val)
    # shuffle(all_fake_images_val)


    # all_real_images_val = []
    # all_real_images_val += sorted(Path("DMDF_Faces_V6/validation/real").glob('**/*.png'))
    # all_real_images_val = list(set(all_real_images_val))
    # all_real_images_val = [str(path) for path in all_real_images_val]
    # all_real_images_val = sorted(all_real_images_val)
    # shuffle(all_real_images_val)
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1
    else:
        print("No checkpoint loaded.")


    print("Model Parameters:", get_n_params(model))
    # print("Total Train Real: " + str(len(all_real_images_train)))
    # print("Total Train Fake: " + str(len(all_fake_images_train)))
    # print("Total Val Real: " + str(len(all_real_images_val)))
    # print("Total Val Fake: " + str(len(all_fake_images_val)))
   
    # #READ DATASET
    # if opt.dataset != "All":
    #     folders = ["Original", opt.dataset]
    # else:
    #     folders = ["Original", "DFDC", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

    # sets = [TRAINING_DIR, VALIDATION_DIR]

    # paths = []
    # for dataset in sets:
    #     for folder in folders:
    #         subfolder = os.path.join(dataset, folder)
    #         for index, video_folder_name in enumerate(os.listdir(subfolder)):
    #             if index == opt.max_videos:
    #                 break
    #             if os.path.isdir(os.path.join(subfolder, video_folder_name)):
    #                 paths.append(os.path.join(subfolder, video_folder_name))
                

    # mgr = Manager()
    # train_dataset = mgr.list()
    # validation_dataset = mgr.list()

    # with Pool(processes=10) as p:
    #     with tqdm(total=len(paths)) as pbar:
    #         for v in p.imap_unordered(partial(read_frames, train_dataset=train_dataset, validation_dataset=validation_dataset),paths):
    #             pbar.update()
    # train_samples = len(train_dataset)
    # train_dataset = shuffle_dataset(train_dataset)
    # validation_samples = len(validation_dataset)
    # validation_dataset = shuffle_dataset(validation_dataset)

    # Print some useful statistics
    # print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    # print("__TRAINING STATS__")
    # train_counters = collections.Counter(image[1] for image in train_dataset)
    # print(train_counters)
    
    # class_weights = train_counters[0] / train_counters[1]
    # print("Weights", class_weights)

    # print("__VALIDATION STATS__")
    # val_counters = collections.Counter(image[1] for image in validation_dataset)
    # print(val_counters)
    # print("___________________")

    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # # Create the data loaders
    # validation_labels = np.asarray([row[1] for row in validation_dataset])
    # labels = np.asarray([row[1] for row in train_dataset])

    # train_dataset = DeepFakesDataset(np.asarray([row[0] for row in train_dataset]), labels, config['model']['image-size'])
    # train_dataset = DeepFakesDataset(all_real_images_train, all_fake_images_train, config['model']['image-size'])
    train_dataset = DeepFakesDataset(dataset_path_labels, config['model']['image-size'])
    
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    del train_dataset

    # validation_dataset = DeepFakesDataset(np.asarray([row[0] for row in validation_dataset]), validation_labels, config['model']['image-size'], mode='validation')
    # validation_dataset = DeepFakesDataset(all_real_images_val, all_fake_images_val, config['model']['image-size'])
    validation_dataset = DeepFakesDataset(dataset_path_labels, config['model']['image-size'], mode='validation')
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    del validation_dataset
    

    model = model.cuda()
    counter = 0
    global_step = 0;
    not_improved_loss = 0
    previous_loss = math.inf
    log_tb_val_steps = 100
    evaluation_steps = 1000
    save_checkpoint_steps = 10000
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
        
        for index, (images, labels, image_path_names) in enumerate(dl):
            model.train()  
            global_step += 1
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.unsqueeze(1)
            images = images.cuda()
            
            y_pred = model(images)
            y_pred = y_pred.cpu()
            loss = loss_fn(y_pred, labels)
        
            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            correct_by_dataset = check_correct_with_paths(y_pred, labels, image_path_names) 
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()
            scheduler.step()
            counter += 1
            total_loss += round(loss.item(), 2)

            if(global_step % log_tb_val_steps):
                writer.add_scalar('Train/CE-VIT Loss: ', loss.item(), global_step)
                for correct_by_dataset_key, correct_by_dataset_value in correct_by_dataset.items():
                    TB_var_name_pos = "Train/CE-ViT Accuracy Positive - " + correct_by_dataset_key
                    TB_var_name_neg = "Train/CE-ViT Accuracy Negative - " + correct_by_dataset_key
                    if((correct_by_dataset_value["total_positive"] > 0)):
                        positive_accuracy = correct_by_dataset_value["correct_positive"]/(correct_by_dataset_value["total_positive"])
                        writer.add_scalar(TB_var_name_pos, positive_accuracy, global_step)
                    if((correct_by_dataset_value["total_negative"] > 0)):
                        negative_accuracy = correct_by_dataset_value["correct_negative"]/(correct_by_dataset_value["total_negative"])
                        writer.add_scalar(TB_var_name_neg, negative_accuracy, global_step)

            for i in range(config['training']['bs']):
                bar.next()

             
            if index%1200 == 0:
                print("\nLoss: ", total_loss/counter, "Accuracy: ",train_correct/(counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive)  

            if global_step % evaluation_steps == 0:
                print("Evaluating data")
                val_counter = 0
                val_correct = 0
                val_positive = 0
                val_negative = 0
                model.eval()
                total_correct_by_dataset = {}
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
                    if not total_correct_by_dataset:
                        total_correct_by_dataset = correct_by_dataset
                    else:
                        for key, val in correct_by_dataset.items():
                            for key2, val2 in val.items():
                                total_correct_by_dataset[key][key2] += val2

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

                writer.add_scalar('Eval/CE-VIT avg Loss: ', total_val_loss, global_step)
                for correct_by_dataset_key, correct_by_dataset_value in total_correct_by_dataset.items():
                    TB_var_name_pos = "Eval/CE-ViT Accuracy Positive - " + correct_by_dataset_key
                    TB_var_name_neg = "Eval/CE-ViT Accuracy Negative - " + correct_by_dataset_key
                    if((correct_by_dataset_value["total_positive"] > 0)):
                        positive_accuracy = correct_by_dataset_value["correct_positive"]/(correct_by_dataset_value["total_positive"])
                        writer.add_scalar(TB_var_name_pos, positive_accuracy, global_step)
                    if((correct_by_dataset_value["total_negative"] > 0)):
                        negative_accuracy = correct_by_dataset_value["correct_negative"]/(correct_by_dataset_value["total_negative"])
                        writer.add_scalar(TB_var_name_neg, negative_accuracy, global_step)

                #print("#" + str(t) + "/" + str(global_step) + " loss:" +
                    #str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + "/" + str(np.count_nonzero(validation_labels == 0)) + " val_1s:" + str(val_positive) + "/" + str(np.count_nonzero(validation_labels == 1)))
    
        
        if(global_step % save_checkpoint_steps):
            if not os.path.exists(MODELS_PATH):
                os.makedirs(MODELS_PATH)
            torch.save(model.state_dict(), os.path.join(MODELS_PATH,  "efficientnet_checkpoint" + str(t) + "_" + opt.dataset))
            
            
