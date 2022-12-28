import cv2
from albumentations import Compose, PadIfNeeded
from transforms_cevit.albu import IsotropicResize
import numpy as np
import os
import cv2
import torch
from statistics import mean
import json
import operator
import random

def transform_frame(image, image_size):
	transform_pipeline = Compose([
				IsotropicResize(max_side=image_size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
				PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_REPLICATE)
				]
			)
	return transform_pipeline(image=image)['image']
	
	
def resize(image, image_size):
	try:
		return cv2.resize(image, dsize=(image_size, image_size))
	except:
		return []

def custom_round(values):
	result = []
	for value in values:
		if value > 0.6:
			result.append(1)
		else:
			result.append(0)
	return np.asarray(result)

	

def get_method(video, data_path):
	methods = os.listdir(os.path.join(data_path, "manipulated_sequences"))
	methods.extend(os.listdir(os.path.join(data_path, "original_sequences")))
	methods.append("DFDC")
	methods.append("Original")
	selected_method = ""
	for method in methods:
		if method in video:
			selected_method = method
			break
	return selected_method

def shuffle_dataset(dataset):
  import random
  random.seed(4)
  random.shuffle(dataset)
  return dataset
  

def get_n_params(model):
	pp=0
	for p in list(model.parameters()):
		nn=1
		for s in list(p.size()):
			nn = nn*s
		pp += nn
	return pp
	
def check_correct(preds, labels):
	preds = preds.cpu()
	labels = labels.cpu()
	preds = [np.asarray(torch.sigmoid(pred).detach().numpy()).round() for pred in preds]

	correct = 0
	correct_positive = 0
	correct_negative = 0
	positive_class = 0
	negative_class = 0
	for i in range(len(labels)):
		pred = int(preds[i])
		if labels[i] == pred:
			correct += 1
		if labels[i] == 1:
			positive_class += 1
			if labels[i] == pred:
				correct_positive += 1
		else:
			negative_class += 1
			if labels[i] == pred:
				correct_negative += 1
	return correct, correct_positive, correct_negative, positive_class, negative_class

def check_correct_with_paths(preds, labels, path_names, dataset_path_labels=["/deeper_forensics/", "/DFDC/", "/Face_Forensics/", "/Face2Face/", "/FaceShifter/", "/FaceSwap/", "/NeuralTextures/", "/deepfacelab/", "/CelebDF/", "/VoxCeleb/", "/FF_GoogleDF/"]):
	preds = preds.cpu()
	labels = labels.cpu()
	preds = [np.asarray(torch.sigmoid(pred).detach().numpy()).round() for pred in preds]
	correct_by_dataset = {}
	for dataset_path_label in dataset_path_labels:
		correct_by_dataset[dataset_path_label] = {
			"total_positive": 0,
			"correct_positive": 0,
			"total_negative": 0,
			"correct_negative": 0
		}
	for i in range(len(labels)):
		pred = int(preds[i])
		path_name = path_names[i]
		proper_key = None
		for dataset_path_label in dataset_path_labels:
			if(dataset_path_label in path_name):
				proper_key = dataset_path_label
		if labels[i] == 1:
			correct_by_dataset[proper_key]["total_positive"] += 1
			if labels[i] == pred:
				correct_by_dataset[proper_key]["correct_positive"] += 1
		else:
			correct_by_dataset[proper_key]["total_negative"] += 1
			if labels[i] == pred:
				correct_by_dataset[proper_key]["correct_negative"] += 1

		# Log demographic information
		final_age, final_gender, final_race, final_emotion = get_demographic_info_from_imagepath(path_name)
		if(final_age is not None):

			# Populate the return dictionary
			age_group = get_age_group_from_number(final_age)

			age_group = "Demographics Predicted AGE " + age_group
			final_gender = "Demographics Predicted GENDER " + final_gender
			final_race = "Demographics Predicted RACE " + final_race
			final_emotion = "Demographics Predicted EMOTION " + final_emotion


			if(age_group not in correct_by_dataset):
				correct_by_dataset[age_group] = {
					"total_positive": 0,
					"correct_positive": 0,
					"total_negative": 0,
					"correct_negative": 0
				}

			if(final_gender not in correct_by_dataset):
				correct_by_dataset[final_gender] = {
					"total_positive": 0,
					"correct_positive": 0,
					"total_negative": 0,
					"correct_negative": 0
				}

			if(final_race not in correct_by_dataset):
				correct_by_dataset[final_race] = {
					"total_positive": 0,
					"correct_positive": 0,
					"total_negative": 0,
					"correct_negative": 0
				}
			
			if(final_emotion not in correct_by_dataset):
				correct_by_dataset[final_emotion] = {
					"total_positive": 0,
					"correct_positive": 0,
					"total_negative": 0,
					"correct_negative": 0
				}

			# Log Age
			if labels[i] == 1:
				correct_by_dataset[age_group]["total_positive"] += 1
				if labels[i] == pred:
					correct_by_dataset[age_group]["correct_positive"] += 1
			else:
				correct_by_dataset[age_group]["total_negative"] += 1
				if labels[i] == pred:
					correct_by_dataset[age_group]["correct_negative"] += 1

			# Log Gender
			if labels[i] == 1:
				correct_by_dataset[final_gender]["total_positive"] += 1
				if labels[i] == pred:
					correct_by_dataset[final_gender]["correct_positive"] += 1
			else:
				correct_by_dataset[final_gender]["total_negative"] += 1
				if labels[i] == pred:
					correct_by_dataset[final_gender]["correct_negative"] += 1

			# Log Race
			if labels[i] == 1:
				correct_by_dataset[final_race]["total_positive"] += 1
				if labels[i] == pred:
					correct_by_dataset[final_race]["correct_positive"] += 1
			else:
				correct_by_dataset[final_race]["total_negative"] += 1
				if labels[i] == pred:
					correct_by_dataset[final_race]["correct_negative"] += 1

			# Log Emotion
			if labels[i] == 1:
				correct_by_dataset[final_emotion]["total_positive"] += 1
				if labels[i] == pred:
					correct_by_dataset[final_emotion]["correct_positive"] += 1
			else:
				correct_by_dataset[final_emotion]["total_negative"] += 1
				if labels[i] == pred:
					correct_by_dataset[final_emotion]["correct_negative"] += 1

	return correct_by_dataset

def custom_video_round(preds):
	for pred_value in preds:
		if pred_value > 0.55:
			return pred_value
	return mean(preds)

def get_age_group_from_number(age):
	if(age < 18):
		return "0-18"
	elif(age < 34):
		return "18-34"
	elif(age < 55):
		return "34-55"
	elif(age < 65):
		return "55-65"
	else:
		return "65+"

def get_dataset_from_imagepath(imagepath, dataset_path_labels=["/deeper_forensics/", "/DFDC/", "/Face_Forensics/", "/Face2Face/", "/FaceShifter/", "/FaceSwap/", "/NeuralTextures/", "/deepfacelab/", "/CelebDF/", "/VoxCeleb/", "/FF_GoogleDF/"]):
	proper_key = None
	for dataset_path_label in dataset_path_labels:
		if(dataset_path_label in imagepath):
			proper_key = dataset_path_label
	return proper_key


def get_demographic_info_from_imagepath(imagepath):
	folderpath = os.path.dirname(imagepath)
	deepface_json_path = folderpath.replace("bounding-box-tight-v2-original-fps/", "deepface/") + "/deepface.json"
	

	if os.path.exists(deepface_json_path):
		with open(deepface_json_path, 'r') as j:
			deepface_dict = json.loads(j.read())

			if(len(deepface_dict.items()) == 0):
				return None, None, None, None

			video_age_total = 0
			video_gender_dict = {}
			video_race_dict = {}
			video_emotion_dict = {}
			for key, value in deepface_dict.items():
				this_age_num = value['age']
				this_gender_val = value['gender']
				this_race_dict = value['race']
				this_emotion_dict = value['emotion']

				video_age_total += this_age_num

				if(this_gender_val in video_gender_dict):
					video_gender_dict[this_gender_val] += 1
				else:
					video_gender_dict[this_gender_val] = 1

				for race_key, race_val in this_race_dict.items():
					if(race_key in video_race_dict):
						video_race_dict[race_key] += race_val
					else:
						video_race_dict[race_key] = race_val

				for emotion_key, emotion_val in this_emotion_dict.items():
					if(emotion_key in video_emotion_dict):
						video_emotion_dict[emotion_key] += emotion_val
					else:
						video_emotion_dict[emotion_key] = emotion_val			

			
			final_age = video_age_total/len(deepface_dict.items())
			final_age = round(final_age, 2)
			final_gender = max(video_gender_dict.items(), key=operator.itemgetter(1))[0]
			final_gender = final_gender.replace("Man", "Masculine").replace("Woman", "Feminine")
			final_race = max(video_race_dict.items(), key=operator.itemgetter(1))[0]
			final_emotion = max(video_emotion_dict.items(), key=operator.itemgetter(1))[0]


			return final_age, final_gender, final_race, final_emotion
	else:
		return None, None, None, None
