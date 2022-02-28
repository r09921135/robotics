import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='RefVOS Training')
    parser.add_argument('--model', default='deeplabv3_resnet101', help='model')
    parser.add_argument('--device', default='cuda', help='device')

    # Fusion language + visual
    parser.add_argument('--multiply_feats', action='store_true', default=True, help='multiplication of visual and language features')
    parser.add_argument('--addition',  action='store_true', help='addition of visual and language features')
    
    # Inference configurations
    parser.add_argument('--checkpoint', default='./refvos/checkpoints/model_best_my_model.pth', help='checkpoint for inference')
    parser.add_argument('--ck_bert',  default='bert-base-uncased', help='BERT pre-trained weights')
    parser.add_argument('--bert_tokenizer',  default='bert-base-uncased', help='BERT tokenizer')

    # Testing image
    parser.add_argument('--test_img', default='./refvos/images/inference/image/0018.png', help='Inference dataset root directory')
    
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args_dict = parser.parse_args()
