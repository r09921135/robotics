import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModelForQuestionAnswering.from_pretrained("./bert/saved_model").to(device)
bert_model.eval()

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x

def actionClassify(input_ids_sentence):
    model = Classifier().to(device)
    model.load_state_dict(torch.load('./bert/Action_Classifier.ckpt'))
    model.eval()
    with torch.no_grad():
        # sentence_tokenized = tokenizer(sentence, add_special_tokens=False)
        # input_ids_sentence = [101] + sentence_tokenized.input_ids + [102]
        input_ids = torch.tensor(input_ids_sentence).unsqueeze(0)
        token_type_ids = torch.tensor([1] * len(input_ids_sentence)).unsqueeze(0)
        attention_mask = torch.tensor([1] * len(input_ids_sentence)).unsqueeze(0)
        output = bert_model.bert(input_ids=input_ids.to(device), token_type_ids=token_type_ids.to(
            device), attention_mask=attention_mask.to(device))
        output = output[0][0][0]
        output = output.unsqueeze(0)
        logits = model(output)
        # give me => label 0
        # feed me => label 1
        action = logits.argmax(dim=-1).cpu().numpy()
        return action


if __name__ == '__main__':
    sentence = 'want to eat'
    action_id = actionClassify(sentence)
    action = 'give' if action_id == 0 else 'feed'
    print(sentence + ' --> ' + action)
