from attrdict import AttrDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import logging
import shutil
import yaml
from data.dataloader import ITM_Dataset
from utils.utils import collate_fn
from models.model_zoo import ViT_NumScore
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def save_checkpoint(state, epoch, dst, is_best):
    filename = os.path.join(dst, str(epoch)) + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        dst_best = os.path.join(dst, 'model_best', str(epoch)) + '.pth.tar'
        shutil.copyfile(filename, dst_best)

def compute_loss(i2t, t2i, labels, batch_size):
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0)
    labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

    i2t_pred = F.softmax(i2t, dim=0)
    i2t_loss = i2t_pred * (F.log_softmax(i2t, dim=0) - torch.log(labels_mask_norm + 1e-6))
    t2i_pred = F.softmax(t2i, dim=0)
    t2i_loss = t2i_pred * (F.log_softmax(t2i, dim=0) - torch.log(labels_mask_norm + 1e-6))

    matching_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))
    return matching_loss

def train(epoch, train_loader, network, optimizer, args, writer):
    network.train()

    train_losses = []

    for step, (images, input_ids, token_type_ids, attention_masks, labels) in enumerate(train_loader):
        with torch.autograd.detect_anomaly():

            images = images.cuda()
            labels = labels.cuda()
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_masks = attention_masks.cuda()
            i2t, t2i = network(images, input_ids, token_type_ids, attention_masks)
            matching_loss = compute_loss(i2t, t2i, labels, args.batch_size)
            train_losses.append(matching_loss.item())
            # writer.add_scalar('training_loss',
            #                   train_losses,
            #                   step)

            if step % 100 == 0:
                print('epoch:{}, step:{}, cmpm_loss:{:.3f}'.format(epoch, step, matching_loss))
            # compute gradient and do ADAM step
            optimizer.zero_grad()
            matching_loss.backward()
            optimizer.step()
    
    train_loss_100 = sum(train_losses[-100:])/100
                
    return train_loss_100

def main(args):
    # prepare dataset
    train_dataset = ITM_Dataset(args.image_root_path,
                                args.sentence_file_path,
                                'train',
                                args.max_length)
    train_loader = data.DataLoader(train_dataset, 
                                   args.batch_size, 
                                   collate_fn=lambda b: collate_fn(b, args.max_length),
                                   shuffle=True, 
                                   num_workers=8,
                                   pin_memory=True,
                                   drop_last=True)
    print('Data loaded')

    network = ViT_NumScore()
    network = nn.DataParallel(network).cuda()
    optimizer = torch.optim.Adam(network.parameters(),
                                 lr = args.lr, 
                                 betas=(args.adam_alpha, args.adam_beta), 
                                 eps=1e-8)
    epoches_list = args.epoches_decay.split('_')
    epoches_list = [int(e) for e in epoches_list]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, epoches_list, 1 - args.lr_decay_ratio)
    print('Network loaded')
    
    writer = SummaryWriter('trained_models/runs/' + args.run_id)
    for i, epoch in enumerate(range(args.epoches)):
        print('Epoch ' + str(i))
        if epoch < epoches_list[0]:
            for param in network.module.image_model.parameters():
                param.requires_grad = False
            for param in network.module.bert.parameters():
                param.requires_grad = False
        elif epoch >= epoches_list[0] and epoch < epoches_list[1]:
            for param in network.module.image_model.parameters():
                param.requires_grad = True
            for param in network.module.bert.parameters():
                param.requires_grad = False
        elif epoch >= epoches_list[1]:
            for param in network.module.image_model.parameters():
                param.requires_grad = True
            for param in network.module.bert.parameters():
                param.requires_grad = True
        # train for one epoch
        train_loss_100 = train(epoch, train_loader, network, optimizer, args, writer)
        # evaluate on validation set
        print('Train done for epoch-{}'.format(epoch))
        
        state = {'network': network.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        save_checkpoint(state, epoch, args.checkpoint_dir, False)
        logging.info('Epoch:  [{}|{}], train_loss: {:.3f}'.format(epoch, args.epoches, train_loss_100))
        scheduler.step()

    logging.info('Train done')
    logging.info(args.checkpoint_dir)
    logging.info(args.log_dir)

if __name__ == "__main__":
    with open('configs.yaml') as f:
        train_args = yaml.load(f, Loader=yaml.FullLoader)['train']
        train_args = AttrDict(train_args)
    main(train_args)
