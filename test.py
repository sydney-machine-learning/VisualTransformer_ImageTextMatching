import os
import logging
import torch
import torch.utils.data as data
from utils.utils import compute_topk, collate_fn
from data.dataloader import ITM_Dataset
import yaml
from attrdict import AttrDict
from models.model_zoo import ViT_NumScore


def test(data_loader, network, args):
    # switch to evaluate mode
    network.eval()
    images_bank = []
    text_bank = []
    labels_bank = []
    index = 0
    with torch.no_grad():
        for images, input_ids, token_type_ids, attention_masks, labels in data_loader:
            images = images.cuda()
            labels = labels.cuda()
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_masks = attention_masks.cuda()

            interval = images.shape[0]
            image_embeddings, text_embeddings = network(images, input_ids, token_type_ids, attention_masks, val=True)
            images_bank.append(image_embeddings)
            text_bank.append(text_embeddings)
            labels_bank.append(labels)
            
            index = index + interval
        

        images_bank = torch.cat(images_bank[:index], dim=0)
        text_bank = torch.cat(text_bank[:index], dim=0)
        labels_bank = torch.cat(labels_bank[:index], dim=0)

        images_bank = images_bank[:1000]
        text_bank = text_bank[:1000]
        labels_bank = labels_bank[:1000]

        scoring_i2t, scoring_t2i = network.scoring_i2t, network.scoring_t2i

        ac_top1_i2t, ac_top10_i2t, ac_top1_t2i, ac_top10_t2i = compute_topk(scoring_i2t, scoring_t2i, images_bank, text_bank, labels_bank, [1,10])
        return ac_top1_i2t, ac_top10_i2t, ac_top1_t2i, ac_top10_t2i


def main(args):
    # prepare dataset
    test_dataset = ITM_Dataset(args.image_root_path,
                                args.sentence_file_path,
                                'test',
                                args.max_length)
    test_loader = data.DataLoader(test_dataset, 
                                   args.batch_size, 
                                   collate_fn=lambda b: collate_fn(b, args.max_length),
                                   shuffle=False, 
                                   num_workers=8,
                                   pin_memory=True,
                                   drop_last=True)
    print('Data loaded')

    ac_i2t_top1_best = 0.0
    ac_i2t_top10_best = 0.0
    ac_t2i_top1_best = 0.0
    ac_t2i_top10_best = 0.0
    i2t_models = os.listdir(args.checkpoint_dir)
    i2t_models.sort()
    for i2t_model in i2t_models:
        model_file = os.path.join(args.checkpoint_dir, i2t_model)
        if os.path.isdir(model_file):
            continue
        epoch = i2t_model.split('.')[0]
        network = ViT_NumScore()
        network = network.cuda()
        network_dict = network.state_dict()
        pretrained_dict = torch.load(model_file)['network']
        # process keyword of pretrained model
        prefix = 'module.image_model.'
        pretrained_dict = {prefix + k[:] :v for k,v in pretrained_dict.items()}
        pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in network_dict}
        network_dict.update(pretrained_dict)
        network.load_state_dict(network_dict)

        ac_top1_i2t, ac_top10_i2t, ac_top1_t2i, ac_top10_t2i = test(test_loader, network, args)
        print(ac_top1_i2t, ac_top10_i2t, ac_top1_t2i, ac_top10_t2i)
        if ac_top1_t2i > ac_t2i_top1_best:
            ac_i2t_top1_best = ac_top1_i2t
            ac_i2t_top10_best = ac_top10_i2t
            ac_t2i_top1_best = ac_top1_t2i
            ac_t2i_top10_best = ac_top10_t2i
            # dst_best = os.path.join(args.checkpoint_dir, 'model_best', str(epoch)) + '.pth.tar'
            # shutil.copyfile(model_file, dst_best)
         
        logging.info('epoch:{}'.format(epoch))
        logging.info('top1_t2i: {:.3f}, top10_t2i: {:.3f}, top1_i2t: {:.3f}, top10_i2t: {:.3f}'.format(
            ac_top1_t2i, ac_top10_t2i, ac_top1_i2t, ac_top10_i2t))
    logging.info('t2i_top1_best: {:.3f}, t2i_top10_best: {:.3f}, i2t_top1_best: {:.3f}, i2t_top10_best: {:.3f}'.format(
            ac_t2i_top1_best, ac_t2i_top10_best, ac_i2t_top1_best, ac_i2t_top10_best))
    logging.info(args.checkpoint_dir)
    logging.info(args.log_dir)

if __name__ == '__main__':
    with open('configs.yaml') as f:
        test_args = yaml.load(f, Loader=yaml.FullLoader)['test']
        test_args = AttrDict(test_args)
    main(test_args)
