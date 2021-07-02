from utils.utils import write_json, makedir
import json
import os
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import pickle

def process_sentences(imgs):
    for img in imgs:
        img['processed_tokens'] = []
        for s in img['sentences']:
            tokens = s['tokens']
            img['processed_tokens'].append(tokens)


def generate_split(args):

    with open(args.json_root,'r') as f:
        imgs = json.load(f)['images']
    # process caption
    process_sentences(imgs)
    train_data = []
    test_data = []
    for img in imgs:
        if img['split'] == 'train':
            train_data.append(img)
        else:
            test_data.append(img)
    train_path = os.path.join(args.out_root, 'train_reid.json')
    test_path = os.path.join(args.out_root, 'test_reid.json')
    write_json(train_data, train_path)
    write_json(test_data, test_path)
    return train_path, test_path


def load_split(args):
    data = []
    splits = ['train', 'test']
    for split in splits:
        split_root = os.path.join(args.out_root, split + '_reid.json')
        with open(split_root, 'r') as f:
            split_data = json.load(f)
        data.append(split_data)
    
    print('load data done')
    return data


def pad(input_tensor, end_value):
    input_list = input_tensor.tolist()[0]
    if len(input_list) > 64:
        input_list = input_list[:64]
    if len(input_list) == 64:
        input_list[-1] = end_value
        return input_list
    else:
        input_list.append(end_value)
        input_list.extend([0]*(64 - len(input_list)))
        return input_list

    
end_values = {'input_ids': 102,
              'token_type_ids': 0,
              'attention_mask': 1}    


def process_data(args):
    if args.first:
        train_path, test_path = generate_split(args)
    else:
        train_path, test_path = load_split(args)
    
    file_names = {'train': {'input': train_path,
                            'output': 'train.pkl'},
                  'test': {'input': test_path,
                           'output': 'test.pkl'}}
    for split in file_names:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        img_paths = []
        labels = []
        token_ids = []
        sentences = []
        with open(os.path.join(args.json_root, file_names[split]['input']), 'r') as f:
            data = json.load(f)
        for img in tqdm(data):
            img_path = img['filename']
            label = img['imgid']
            for sentence in img['sentences']:
                img_paths.append(img_path)
                labels.append(label)
                sentences.append(sentence)
                sentence_raw = sentence['raw']
                encoded_input = tokenizer(sentence_raw, padding=True, truncation=True, max_length=64, return_tensors='pt')
                for k in encoded_input:
                    encoded_input[k] = pad(encoded_input[k], end_values[k])
                    
                token_ids.append(encoded_input)
        with open(os.path.join(args.out_root, "bert_tokenizer_" + file_names[split]['output']), "wb") as fOut:
            pickle.dump({'images_path': img_paths,
                        'labels': labels,
                        'sentences': sentences, 
                        'token_ids': token_ids
                        }, 
                        fOut, 
                        protocol=pickle.HIGHEST_PROTOCOL) 


def parse_args():
    parser = argparse.ArgumentParser(description='Command for data preprocessing')
    parser.add_argument('--json_root', type=str)
    parser.add_argument('--out_root',type=str)
    parser.add_argument('--min_word_count', type=int)
    parser.add_argument('--default_image_size', type=int, default=224)
    parser.add_argument('--first', action='store_true')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    makedir(args.out_root)
    process_data(args)