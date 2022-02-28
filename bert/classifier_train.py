import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import pandas as pd
from tqdm.auto import tqdm
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fix random seed for reproducibility


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


same_seeds(0)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModelForQuestionAnswering.from_pretrained("./saved_model").to(device)
bert_model.eval()
for param in bert_model.parameters():
    param.requires_grad = False


class MyDataset(Dataset):
    def __init__(self, train_sentences_tokenized, train_labels, bert_model):
        self.bert_model = bert_model
        self.train_sentences_tokenized = train_sentences_tokenized
        self.train_labels = train_labels
        self.max_seq_len = 16

    def __len__(self):
        return len(train_labels)

    def __getitem__(self, idx):
        input_ids_sentence = [
            101] + self.train_sentences_tokenized[idx].input_ids + [102]
        padding_len = self.max_seq_len - len(input_ids_sentence)
        input_ids = torch.tensor(
            input_ids_sentence + [0] * padding_len).unsqueeze(0)
        token_type_ids = torch.tensor(
            [1] * len(input_ids_sentence) + [0] * padding_len).unsqueeze(0)
        attention_mask = torch.tensor(
            [1] * len(input_ids_sentence) + [0] * padding_len).unsqueeze(0)
        features = self.bert_model(input_ids=input_ids.to(device), token_type_ids=token_type_ids.to(
            device), attention_mask=attention_mask.to(device))
        # features = bert_model(input_ids=input_ids.to(device))
        label = self.train_labels[idx]
        # print(features.shape, label)
        return features[0][0][0], label


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


def read_data(file):
    pred = pd.read_csv(file)
    return pred['Sentence'], pred['label']


train_sentences, train_labels = read_data("classifier_train.csv")
train_sentences_tokenized = [
    tokenizer(sentence, add_special_tokens=False) for sentence in train_sentences]

train_set = MyDataset(train_sentences_tokenized, train_labels, bert_model.bert)

train_batch_size = 3

train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)

model = Classifier().to(device)
num_epoch = 100
learning_rate = 3e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()
criterion = nn.CrossEntropyLoss()
print("Start Training ...")

train_acc = 0
for epoch in range(num_epoch):
    train_loss = []
    train_accs = []
    for (features, labels) in tqdm(train_loader):
        features, labels = features.to(device), labels.to(device)
        logits = model(features.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)
        # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print(
        f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    torch.save(model.state_dict(), 'Action_Classifier.ckpt')
