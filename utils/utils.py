import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import json
import torch
from tqdm import tqdm

# image transform 
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def makedir(root):
    if not os.path.exists(root):
        os.makedirs(root)


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)


def collate_fn(data, max_length):
    batch_size = len(data)
    _, image_c, image_h, image_w = data[0][0].shape
    images = torch.zeros(batch_size, image_c, image_h, image_w).float()
    # captions = torch.zeros(batch_size, text_embedding_size).float()
    input_ids = torch.zeros(batch_size, max_length).long()
    token_type_ids = torch.zeros(batch_size, max_length).long()
    attention_masks = torch.zeros(batch_size, max_length).long()
    labels = torch.zeros(batch_size).long()
    for i in range(batch_size):
        images[i] = data[i][0]
        # captions[i] = torch.from_numpy(data[i][1])
        input_ids[i] = torch.LongTensor(data[i][1])
        token_type_ids[i] = torch.LongTensor(data[i][2])
        attention_masks[i] = torch.LongTensor(data[i][3])
        labels[i] = data[i][4]
    return images, input_ids, token_type_ids, attention_masks, labels


def compute_topk(scoring_i2t, scoring_t2i, images_embeddings, text_embeddings, labels, k=[1, 10]):
    images_embeddings_norm = images_embeddings/images_embeddings.norm(dim=2)[:, :, None]
    text_embeddings_norm = text_embeddings/text_embeddings.norm(dim=1)[:, None]
    batch_size = images_embeddings.shape[0]
    i2t = []
    t2i = []
    for i in tqdm(range(batch_size)):
        item_i2t = torch.matmul(images_embeddings[i, :, :]. unsqueeze(0), text_embeddings_norm.transpose(0, 1))
        item_t2i = torch.matmul(images_embeddings_norm[i, :, :].unsqueeze(0), text_embeddings.transpose(0, 1))

        item_i2t, item_t2i = item_i2t.transpose(1, 2), item_t2i.transpose(1, 2)
        item_i2t = scoring_i2t(item_i2t).squeeze().unsqueeze(0)
        item_t2i = scoring_t2i(item_t2i).squeeze(-1)

        i2t.append(item_i2t)
        t2i.append(item_t2i)
    # i2t = torch.matmul(images_embeddings, text_embeddings_norm.transpose(0, 1))
    # t2i = torch.matmul(images_embeddings_norm, text_embeddings.transpose(0, 1))
    i2t = torch.cat(i2t, dim=0)
    t2i = torch.cat(t2i, dim=0)
    t2i = t2i.transpose(0, 1)
    # i2t, t2i = i2t.transpose(1, 2), t2i.transpose(1, 2)
    # i2t = scoring_i2t(i2t).squeeze()
    # t2i = scoring_t2i(t2i).squeeze().transpose(0, 1)

    result = []
    result.extend(topk(i2t, labels, k=[1, 10]))
    result.extend(topk(t2i, labels, k=[1, 10]))
    return result

def topk(sim, labels, k=[1, 10]):
    result = []
    maxk = max(k)
    size_total = len(labels)
    _, pred_index = sim.topk(maxk, 0, True, True)
    pred_labels = labels[pred_index]
    correct = pred_labels.eq(labels.view(1,-1).expand_as(pred_labels))
    print(labels)
    for topk in k:
        correct_k = torch.sum(correct[:topk], dim=0)
        correct_k = torch.sum(correct_k > 0).float()
        result.append(correct_k * 100 / size_total)
        print(correct_k, size_total)
    return result
