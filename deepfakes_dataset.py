import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np
import random
import uuid
import albumentations as A

from transforms_cevit.albu import IsotropicResize
from pathlib import Path


MAXIMUM_VALIDATION_SIZE = 10000
class DeepFakesDataset(Dataset):
    # def __init__(self, images, labels, image_size, mode = 'train'):
    
    def __init__(self, dataset_path_labels_fake, dataset_path_labels_real, image_size, mode = 'train', max_images=0):
        self.dataset_root_name = 'dmdf_v2/'
        self.crop_type = "bounding-box-tight-v2-original-fps/"

        # Get a dictionary of image paths split by each dataset type
        fake_paths_dictionary, real_paths_dictionary = self.get_all_images(dataset_path_labels_fake, dataset_path_labels_real, mode)
        self.fake_paths_dictionary = fake_paths_dictionary
        self.real_paths_dictionary = real_paths_dictionary
        self.image_size = image_size
        self.mode = mode

        # Calculate total amount of real and fake images across all datasets
        total_fakes = 0
        total_reals = 0
        for fake_dataset_key, fake_dataset_value in fake_paths_dictionary.items():
            total_fakes += len(fake_dataset_value)

        for real_dataset_key, real_dataset_value in real_paths_dictionary.items():
            total_reals += len(real_dataset_value)

        print("Loaded Image Paths: " + mode)
        print("Fake: " + str(total_fakes))
        print("Real: " + str(total_reals))

        # Reduce validation size to maximum value
        if(max_images == 0):
            self.n_samples = total_fakes + total_reals
        else:
            self.n_samples = min(MAXIMUM_VALIDATION_SIZE, (total_fakes + total_reals))

        print("Number Samples: " + str(self.n_samples))
        

    def get_all_images(self, dataset_path_labels_fake, dataset_path_labels_real, split):

        reals_dataset_dictionary = {}
        fakes_dataset_dictionary = {}

        # Get all images for each dataset type as separate lists for real and fake
        # Create a list of lists for real and fake
        for dataset_path_label in dataset_path_labels_fake:

            all_fake_images_train = []
            all_fake_images_train += sorted(Path(self.dataset_root_name + split + "/fake" + dataset_path_label + self.crop_type).glob('**/*.png'))
            all_fake_images_train = list(set(all_fake_images_train))
            all_fake_images_train = [str(path) for path in all_fake_images_train]
            all_fake_images_train = sorted(all_fake_images_train)
            
            print("Dataset Fakes")
            print(dataset_path_label)
            print("FAKES: " + str(len(all_fake_images_train)))
            fakes_dataset_dictionary[dataset_path_label] = all_fake_images_train

        for dataset_path_label in dataset_path_labels_real:

            all_real_images_train = []
            all_real_images_train += sorted(Path(self.dataset_root_name + split + "/real" + dataset_path_label + self.crop_type).glob('**/*.png'))
            all_real_images_train = list(set(all_real_images_train))
            all_real_images_train = [str(path) for path in all_real_images_train]
            all_real_images_train = sorted(all_real_images_train)
            
            print("Dataset Reals")
            print(dataset_path_label)
            print("REALS: " + str(len(all_real_images_train)))
            reals_dataset_dictionary[dataset_path_label] = all_real_images_train

        return fakes_dataset_dictionary, reals_dataset_dictionary


    
    # Augmentations
    def create_train_transforms(self, size):
        return A.Compose([
            A.Affine(p=0.10, scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=(-3, 3), shear=(-3, 3), mode=cv2.BORDER_REPLICATE),
            A.CLAHE(always_apply=False, p=0.10, clip_limit=(1, 4), tile_grid_size=(8, 8)),
            A.CoarseDropout(always_apply=False, p=0.10, max_holes=8, max_height=8, max_width=8, min_holes=2, min_height=2, min_width=2),
            A.ChannelShuffle(always_apply=False, p=0.05),
            A.GaussNoise(always_apply=False, p=0.10, var_limit=(10.0, 50.0)),
            A.HueSaturationValue(always_apply=False, p=0.3, hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20)),
            A.ImageCompression(always_apply=False, p=0.05, quality_lower=60, quality_upper=100, compression_type=0),
            A.MotionBlur(always_apply=False, p=0.05, blur_limit=(3, 7)),
            A.RandomBrightnessContrast(always_apply=False, p=0.3, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
            A.HorizontalFlip(p=0.5),
            A.ToGray(p=0.05),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5)

            # ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            # GaussNoise(p=0.3),
            # GaussianBlur(blur_limit=3, p=0.05),
            # HorizontalFlip(),
            # PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            # OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.4),
            # ToGray(p=0.2),
            # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        )

    # No Augmentations
    # def create_train_transforms(self, size):
    #     return Compose([
    #         Resize(height=size, width=size, interpolation=cv2.INTER_CUBIC, p=1.0),
    #     ])
        
    # def create_val_transform(self, size):
    #     return Compose([
    #         Resize(height=size, width=size, interpolation=cv2.INTER_CUBIC, p=1.0),
    #     ])

    def __getitem__(self, index):


        is_fake_image = random.choice([True, False])

        if(is_fake_image):
            dataset_type = random.choice(list(self.fake_paths_dictionary.values()))
            while len(dataset_type) == 0:
                dataset_type = random.choice(list(self.fake_paths_dictionary.values()))
            this_image_path = random.choice(dataset_type)
            label = 1.
        else:
            dataset_type = random.choice(list(self.real_paths_dictionary.values()))
            while len(dataset_type) == 0:
                dataset_type = random.choice(list(self.real_paths_dictionary.values()))
            this_image_path = random.choice(dataset_type)
            label = 0.

        image = cv2.imread(str(this_image_path))
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation = cv2.INTER_CUBIC)

        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size)
            image = transform(image=image)['image']
        # else:
        #     transform = self.create_val_transform(self.image_size)


        #unique = uuid.uuid4()
        #cv2.imwrite("../dataset/augmented_frames/vit_augmentation/square_fda/"+str(unique)+"_"+str(index)+"_original.png", image)
   
        # image = transform(image=image)['image']

        image = image/255.0
        
        #cv2.imwrite("../dataset/augmented_frames/vit_augmentation/square_fda/"+str(unique)+"_"+str(index)+".png", image)
        
        return torch.tensor(image).float(), label, this_image_path



    def __len__(self):
        return self.n_samples

 
