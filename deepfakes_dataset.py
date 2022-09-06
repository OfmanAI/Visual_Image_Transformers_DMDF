import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np
import random
import uuid
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, Rotate

from transforms.albu import IsotropicResize
from pathlib import Path

class DeepFakesDataset(Dataset):
    # def __init__(self, images, labels, image_size, mode = 'train'):
    
    def __init__(self, dataset_path_labels, image_size, mode = 'train'):
        # self.x = images
        # self.y = torch.from_numpy(labels)
        fake_paths, real_paths = self.get_all_images(dataset_path_labels, mode, should_normalize_presence=False)
        print("Loaded Image Paths: " + mode)
        print("Fake: " + str(len(fake_paths)))
        print("Real: " + str(len(real_paths)))
        self.fake_paths = fake_paths
        self.real_paths = real_paths
        self.image_size = image_size
        self.mode = mode
        self.n_samples = len(self.fake_paths) + len(self.real_paths)
        

    def get_all_images(self, dataset_path_labels, split, should_normalize_presence=False):
        reals_dataset_list = []
        fakes_dataset_list = []
        max_in_dataset = 100000
        

        # Get all images for each dataset type as separate lists for real and fake
        # Create a list of lists for real and fake
        for dataset_path_label in dataset_path_labels:

            all_fake_images_train = []
            all_fake_images_train += sorted(Path("DMDF_Faces_V6/" + split + "/fake" + dataset_path_label).glob('**/*.png'))
            all_fake_images_train = list(set(all_fake_images_train))
            all_fake_images_train = [str(path) for path in all_fake_images_train]
            all_fake_images_train = sorted(all_fake_images_train)
            fakes_dataset_list.append(all_fake_images_train)

            all_real_images_train = []
            all_real_images_train += sorted(Path("DMDF_Faces_V6/" + split + "/real" + dataset_path_label).glob('**/*.png'))
            all_real_images_train = list(set(all_real_images_train))
            all_real_images_train = [str(path) for path in all_real_images_train]
            all_real_images_train = sorted(all_real_images_train)
            reals_dataset_list.append(all_real_images_train)
            print(dataset_path_label)

        # Loop through each dataset real and fake to find the maximum across all datasets
        max_length_real = 0
        for reals_dataset in reals_dataset_list:
            if(len(reals_dataset) > max_length_real):
                max_length_real = len(reals_dataset)

        max_length_fake = 0
        for fakes_dataset in fakes_dataset_list:
            if(len(fakes_dataset) > max_length_fake):
                max_length_fake = len(fakes_dataset)

        # Loop through each real and fake dataset, upsample images so each dataset has the same number of images, normalized dataset type presence during training
        final_real_images_set = []
        max_length_real = min(max_length_real, max_in_dataset)
        for reals_dataset in reals_dataset_list:
            if(len(reals_dataset) == 0):
                continue
            if(should_normalize_presence == False):
                this_upsampled_reals_dataset = reals_dataset
            else:
                this_upsampled_reals_dataset = np.random.choice(reals_dataset, max_length_real)
            final_real_images_set.extend(this_upsampled_reals_dataset)

        final_fake_images_set = []
        max_length_fake = min(max_length_fake, max_in_dataset)
        for fakes_dataset in fakes_dataset_list: 
            if(len(fakes_dataset) == 0):
                continue
            if(should_normalize_presence == False):
                this_upsampled_fakes_dataset = fakes_dataset
            else:
                this_upsampled_fakes_dataset = np.random.choice(fakes_dataset, max_length_fake)
            final_fake_images_set.extend(this_upsampled_fakes_dataset)

        return final_fake_images_set, final_real_images_set

    
    def create_train_transforms(self, size):
        return Compose([
            ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
            GaussNoise(p=0.3),
            #GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            OneOf([
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p=1),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.4),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        )
        
    def create_val_transform(self, size):
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ])

    def __getitem__(self, index):


        is_fake_image = random.choice([True, False])

        if(is_fake_image):
            this_image_path = random.choice(self.fake_paths)
            label = 1.
        else:
            this_image_path = random.choice(self.real_paths)
            label = 0.

        image = cv2.imread(str(this_image_path))

        
        # image = np.asarray(self.x[index])


        
        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size)
        else:
            transform = self.create_val_transform(self.image_size)
                
        #unique = uuid.uuid4()
        #cv2.imwrite("../dataset/augmented_frames/vit_augmentation/square_fda/"+str(unique)+"_"+str(index)+"_original.png", image)
   
        image = transform(image=image)['image']
        
        #cv2.imwrite("../dataset/augmented_frames/vit_augmentation/square_fda/"+str(unique)+"_"+str(index)+".png", image)
        
        return torch.tensor(image).float(), label, this_image_path



    def __len__(self):
        return self.n_samples

 
