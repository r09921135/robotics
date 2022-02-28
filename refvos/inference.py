import matplotlib.pyplot as plt
import os

import torch
import torch.utils.data
from torch import nn

from transformers import *
import torchvision

from lib import segmentation
from data.dataset_refer_bert import InferenceDataset

from torchvision import transforms
import utils

import numpy as np

from imageio import imread



def evaluate(args, model, data_loader, bert_model, device, display=False,):
    model.eval()
    with torch.no_grad():
        for image, sentences, attentions, name in data_loader:
            # image: (B, C, H, W)
            # sentences: (B, token_len)
            # attentions(masks): (B, token_len)
            image, sentences, attentions = image.to(device), sentences.to(device), attentions.to(device)

            # last_hidden_states: (B, token_len, 768)
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
            
            embedding = last_hidden_states[:, 0, :]

            # output: (B, 2, H, W)
            output, _, _ = model(image, embedding.squeeze(1))

            output = output['out'].cpu()
            output_mask = output.argmax(1).data.numpy()

            if display:

                plt.figure()
                plt.axis('off')
                
                im = plt.imread(os.path.join(args.inference_data_root, 'image',name[0]))
                plt.imshow(im)

                ax = plt.gca()
                ax.set_autoscale_on(False)

                # mask definition
                img = np.ones((im.shape[0], im.shape[1], 3))
                color_mask = np.array([0, 255, 0]) / 255.0
                for i in range(3):
                    img[:, :, i] = color_mask[i]

                output_mask = output_mask.transpose(1, 2, 0)

                ax.imshow(np.dstack((img, output_mask * 0.5)))

                if not os.path.isdir(args.results_folder):
                    os.makedirs(args.results_folder)

                figname = os.path.join(args.results_folder, (name[0].split('.'))[0] + '.png')
                plt.savefig(figname)
                # plt.show()
                plt.close()

    return output_mask


def get_transform():
    
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    return tfm


def main(args):

    device = torch.device(args.device)

    # data sample: image, target, sentences, attentions
    # image: (C, H, W)
    # sentences: (token_len, 3)
    # attentions(masks): (token_len, 3)
    dataset_test = InferenceDataset(args, get_transform())
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1)

    model = segmentation.__dict__[args.model](num_classes=2,
        aux_loss=False,
        pretrained=False,
        args=args)

    model.to(device)
    model_class = BertModel

    bert_model = model_class.from_pretrained(args.ck_bert)
    bert_model.to(device)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    bert_model.load_state_dict(checkpoint['bert_model'])
    model.load_state_dict(checkpoint['model'])

    outputs = evaluate(args, model, data_loader_test, bert_model, device=device, display=True)



if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)