import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random

import transformers

import h5py

from datasets.refer.refer import REFER

from args import get_parser

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 input_size,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.input_size = input_size
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)

        self.max_tokens = 20

        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids

        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = transformers.BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode

        pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[PAD]'))

        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['raw']  # e.g lower left corner darkness 
                attention_mask = [0] * self.max_tokens
                padded_input_ids = [0] * self.max_tokens

                input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)  # e.g [101, 2896, 2187, 3420, 4768, 102]

                # truncation of tokens
                input_ids = input_ids[:self.max_tokens]

                padded_input_ids[:len(input_ids)] = input_ids
                attention_mask[:len(input_ids)] = [1]*len(input_ids)

                sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))

            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)
       

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_ids)

    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")

        ref = self.refer.loadRefs(this_ref_id)
        this_sent_ids = ref[0]['sent_ids']

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])

        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # involves transform from PIL to tensor and mean and std normalization
            img, target = self.image_transforms(img, annot)


        if self.eval_mode:

            embedding = []
            att = []
            for s in range(len(self.input_ids[index])):
                e = self.input_ids[index][s]
                a = self.attention_masks[index][s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)

        else:

            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]

        return img, target, tensor_embeddings, attention_mask



class FineTuneDataset(data.Dataset):
    def __init__(self, args, image_transform=None, target_transform=None):
        """ Intialize the dataset """
        self.path = args.finetune_data_root
        self.image_transform = image_transform
        self.target_transform = target_transform

        # read filenames
        image_fn = [filename for filename in os.listdir(os.path.join(self.path, 'image'))]
        self.image_fn = sorted(image_fn)
        target_fn = [filename for filename in os.listdir(os.path.join(self.path, 'mask'))]
        self.target_fn = sorted(target_fn)
        sentence_fn = [filename for filename in os.listdir(os.path.join(self.path, 'text'))]
        self.sentence_fn = sorted(sentence_fn)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        max_tokens = 20
        angle = angle = random.randint(-25, 25)
        attention_mask = [0] * max_tokens
        padded_token = [0] * max_tokens

        image_fn = self.image_fn[index]
        image = Image.open(os.path.join(self.path, 'image', image_fn)).convert('RGB')
        if self.image_transform is not None:
            image = self.image_transform(image)
            image = TF.rotate(image, angle)

        target_fn = self.target_fn[index]
        target = Image.open(os.path.join(self.path, 'mask', target_fn))
        if self.target_transform is not None:
            target = self.target_transform(target)
            target = TF.rotate(target, angle)

        sentence_fn = self.sentence_fn[index]
        f = open(os.path.join(self.path, 'text', sentence_fn), 'r')
        lines = f.readlines()
        sentence = lines[0]
        
        tokenizer = transformers.BertTokenizer.from_pretrained(args.bert_tokenizer)
        token = tokenizer.encode(text=sentence, add_special_tokens=True)
        padded_token[:len(token)] = token
        padded_token = torch.tensor(padded_token)
        attention_mask[:len(token)] = [1]*len(token)
        attention_mask = torch.tensor(attention_mask)

        return image, target[0], padded_token.unsqueeze(0), attention_mask.unsqueeze(0)

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.image_fn)



class InferenceDataset(data.Dataset):
    def __init__(self, args, transform=None):
        """ Intialize the dataset """
        self.path = args.inference_data_root
        self.transform = transform

        # read filenames
        image_fn = [filename for filename in os.listdir(os.path.join(self.path, 'image'))]
        self.image_fn = sorted(image_fn)
        sentence_fn = [filename for filename in os.listdir(os.path.join(self.path, 'text'))]
        self.sentence_fn = sorted(sentence_fn)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        max_tokens = 20
        attention_mask = [0] * max_tokens
        padded_token = [0] * max_tokens

        image_fn = self.image_fn[index]
        image = Image.open(os.path.join(self.path, 'image', image_fn)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        sentence_fn = self.sentence_fn[index]
        f = open(os.path.join(self.path, 'text', sentence_fn), 'r')
        lines = f.readlines()
        sentence = lines[0]
        
        tokenizer = transformers.BertTokenizer.from_pretrained(args.bert_tokenizer)
        token = tokenizer.encode(text=sentence, add_special_tokens=True)
        padded_token[:len(token)] = token
        padded_token = torch.tensor(padded_token)
        attention_mask[:len(token)] = [1]*len(token)
        attention_mask = torch.tensor(attention_mask)

        return image, padded_token, attention_mask ,image_fn

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.image_fn)



if __name__ == '__main__':

    tfm_image = transforms.Compose([
        transforms.Resize(480),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    tfm_target = transforms.Compose([
        transforms.Resize(480, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ]) 

    dataset = FineTuneDataset(args, tfm_image, tfm_target)