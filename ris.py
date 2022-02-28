import torch
from torchvision import transforms
from transformers import *

from ris_lib import segmentation



def preprocess(image, token):
    max_tokens = 20
    padded_token = [0] * max_tokens
    attention_mask = [0] * max_tokens

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    image = tfm(image)
    
    padded_token[:len(token)] = token
    padded_token = torch.tensor(padded_token)

    attention_mask[:len(token)] = [1]*len(token)
    attention_mask = torch.tensor(attention_mask)

    return image.unsqueeze(0), padded_token.unsqueeze(0), attention_mask.unsqueeze(0)


def RIS(args, image, token):
    device = torch.device(args.device)

    model = segmentation.__dict__[args.model](num_classes=2,
        aux_loss=False,
        pretrained=False,
        args=args)

    model.to(device)
    model_class = BertModel

    bert_model = model_class.from_pretrained(args.ck_bert)
    bert_model.to(device)

    checkpoint = torch.load(args.checkpoint)

    bert_model.load_state_dict(checkpoint['bert_model'])
    model.load_state_dict(checkpoint['model'])

    model.eval()
    with torch.no_grad():
        # image: (B, C, H, W)
        # sentences: (B, token_len)
        # attentions(masks): (B, token_len)
        image, sentences, attentions = preprocess(image, token)
        image, sentences, attentions = image.to(device), sentences.to(device), attentions.to(device)

        last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]        
        embedding = last_hidden_states[:, 0, :]

        output, _, _ = model(image, embedding.squeeze(1))

        output = output['out'].cpu()
        output_mask = output.argmax(1).data.numpy()

    return output_mask



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    RIS(args)