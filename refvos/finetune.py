import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from functools import reduce #python 3
import operator

from transformers import *
from PIL import Image
import torchvision

from lib import segmentation

from coco_utils import get_coco
from torchvision import transforms
import utils

import numpy as np

import gc

from data.dataset_refer_bert import FineTuneDataset



def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr - args.lr_specific_decrease*epoch
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# IoU calculation for proper validation
def IoU(pred, gt):

    pred = pred.argmax(1)

    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection

    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)

    return iou


def criterion(inputs, target, args):
    losses = {}
    for name, x in inputs.items():

        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, args, bert_model, device, num_classes, epoch):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    val_loss = 0
    seg_loss = 0
    cos_loss = 0
    total_its = 0

    acc_ious = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):

            total_its += 1

            # image: (B, C, H, W)
            # target: (B, H, W)
            # sentences: (B, 1, token_len)
            # attentions(masks): (B, 1, token_len)
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device).long(), sentences.to(
                        device), attentions.to(device)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]

            embedding = last_hidden_states[:, 0, :]
            output, vis_emb, lan_emb = model(image, embedding.squeeze(1))

            iou = IoU(output['out'], target)
            acc_ious += iou

            loss = criterion(output, target, args)

            output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

        val_loss = val_loss/total_its
        iou = acc_ious / total_its

    return confmat, iou


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args, print_freq,
                    iterations, bert_model):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0
    train_emb_loss = 0
    train_seg_loss = 0

    for data in metric_logger.log_every(data_loader, print_freq, header):
       
        total_its += 1
    
        # image: (B, C, H, W)
        # target: (B, H, W)
        # sentences: (B, 1, token_len)
        # attentions(masks): (B, 1, token_len)
        image, target, sentences, attentions = data
        image, target, sentences, attentions = image.to(device), target.to(device).long(), sentences.to(device), attentions.to(device)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
             
        embedding = last_hidden_states[:, 0, :]
        output, vis_emb, lan_emb = model(image, embedding.squeeze(1))

        loss = criterion(output, target, args)
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.linear_lr:
            adjust_learning_rate(optimizer, epoch, args)
        else:
            lr_scheduler.step()

        train_loss += loss.item()
        iterations += 1

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, embedding, output, vis_emb, lan_emb, last_hidden_states, data

        gc.collect()
        torch.cuda.empty_cache()

    train_loss = train_loss/total_its


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    device = torch.device(args.device)
    num_classes = 2

    tfm_image = transforms.Compose([
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tfm_target = transforms.Compose([
        transforms.Resize((480, 480), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ]) 

    # data sample: image, target, sentences, attentions
    # image: (C, H, W)
    # target: (H, W)
    # sentences: (1, token_len)
    # attentions(masks): (1, token_len)
    dataset = FineTuneDataset(args, tfm_image, tfm_target)
    dataset_test = FineTuneDataset(args, tfm_image, tfm_target)

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn_emb_berts, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn_emb_berts)

    model = segmentation.__dict__[args.model](num_classes=num_classes,
        aux_loss=args.aux_loss,
        pretrained=args.pretrained,
        args=args)

    model_class = BertModel
    bert_model = model_class.from_pretrained(args.ck_bert)

    # load pretrained model
    if args.pretrained_refvos:
        checkpoint = torch.load(args.ck_pretrained_refvos)
        model.load_state_dict(checkpoint['model'])
        bert_model.load_state_dict(checkpoint['bert_model'])

    elif args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model = model.cuda()
    bert_model = bert_model.cuda()

    model_without_ddp = model
    bert_model_without_ddp = bert_model

    if args.test_only:
        confmat = evaluate(model, data_loader_test, args, bert_model, epoch=0, device=device, num_classes=num_classes)
        print(confmat)
        return

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
        # the following are the parameters of bert
        {"params": reduce(operator.concat, [[p for p in bert_model_without_ddp.encoder.layer[i].parameters() if p.requires_grad] for i in range(10)])},
        {"params": [p for p in bert_model_without_ddp.pooler.parameters() if p.requires_grad]}
    ]

    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.fixed_lr:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: args.lr_specific)
    elif args.linear_lr:
        lr_scheduler = None
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    model_dir = os.path.join('./models/', args.model_id)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
        os.makedirs(os.path.join(model_dir, 'train'))
        os.makedirs(os.path.join(model_dir, 'val'))

    start_time = time.time()

    iterations = 0
    t_iou = 0

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

        if not args.fixed_lr:
            if not args.linear_lr:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])


    for epoch in range(args.epochs):
        
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args, args.print_freq,
                        iterations, bert_model)

        confmat, iou = evaluate(model, data_loader_test, args, bert_model, epoch=epoch, device=device,
                                num_classes=num_classes)

        # display evaluation results
        print(confmat)

        # only save if checkpoint improves
        if t_iou < iou:
            print('Better epoch: {}\n'.format(epoch))

            dict_to_save = {'model': model_without_ddp.state_dict(),
            'bert_model': bert_model.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            # 'epoch': epoch,
            # 'args': args
            }

            if not args.linear_lr:
                dict_to_save['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(dict_to_save, os.path.join(args.output_dir, 'model_best_{}.pth'.format(args.model_id)))

            t_iou = iou

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()

    main(args)

