import os
import torch.utils.data as data
import pickle
import torchvision.transforms as T
from utils.utils import train_transform, test_transform
from PIL import Image
from transformers import ViTFeatureExtractor


class ITM_Dataset(data.Dataset):
    def __init__(self,
                 image_root_path: str,
                 sentence_file_path: str,
                 split: str,
                 max_length: int):
        self.image_root_path = image_root_path
        self.sentence_file_path = sentence_file_path
        self.split = split
        self.max_length = max_length

        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

        with open(self.sentence_file_path, 'rb') as f:
            data = pickle.load(f)
            self.img_labels = data['labels']
            self.sentences = data['token_ids']
            self.img_paths = data['images_path']
        
    def __len__(self):
        return len(self.img_labels)
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        sentence = self.sentences[index]
        img_label = self.img_labels[index]

        img = Image.open(os.path.join(self.image_root_path, img_path))
        img = self.feature_extractor(img, return_tensors="pt")['pixel_values']

        input_id = sentence['input_ids']
        token_type_id = sentence['token_type_ids']
        attention_mask = sentence['attention_mask']

        return img, input_id, token_type_id, attention_mask, img_label
