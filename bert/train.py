import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, AutoModelForQuestionAnswering, AutoTokenizer
import pandas as pd
from tqdm.auto import tqdm

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

model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

"""# Read CSV file"""


def read_data(file):
    pred = pd.read_csv(file)
    return pred['Question'], pred['Sentence'], pred['Answer'], pred['start'], pred['end']


train_questions, train_sentences, train_answers, train_start, train_end = read_data("train.csv")
test_questions, test_sentences, test_answers, test_start, test_end = read_data("test.csv")

train_questions_tokenized = [
    tokenizer(question, add_special_tokens=False) for question in train_questions]
test_questions_tokenized = [
    tokenizer(question, add_special_tokens=False) for question in test_questions]

train_sentences_tokenized = [
    tokenizer(sentence, add_special_tokens=False) for sentence in train_sentences]
test_sentences_tokenized = [
    tokenizer(sentence, add_special_tokens=False) for sentence in test_sentences]

"""# Dataset"""


class QA_Dataset(Dataset):
    def __init__(self, tokenized_questions, tokenized_sentences, start_indexs, end_indexs):
        self.tokenized_questions = tokenized_questions
        self.tokenized_sentences = tokenized_sentences
        self.start_indexs = start_indexs
        self.end_indexs = end_indexs
        self.max_seq_len = 32

    def __len__(self):
        return len(self.tokenized_questions)

    def __getitem__(self, idx):
        # print(self.tokenized_questions[idx])
        input_ids_question = [101] + \
            self.tokenized_questions[idx].input_ids + [102]
        input_ids_sentence = self.tokenized_sentences[idx].input_ids + [102]

        padding_len = self.max_seq_len - \
            len(input_ids_question) - len(input_ids_sentence)
        input_ids = torch.tensor(
            input_ids_question + input_ids_sentence + [0] * padding_len)
        # print(input_ids[0])
        token_type_ids = torch.tensor(
            [0] * len(input_ids_question) + [1] * len(input_ids_sentence) + [0] * padding_len)
        # print(token_type_ids.shape)
        attention_mask = torch.tensor(
            [1] * len(input_ids_question) + [1] * len(input_ids_sentence) + [0] * padding_len)
        # Convert answer's start/end positions in paragraph_text to start/end positions in tokenized_paragraph
        answer_start_token = input_ids[int(self.start_indexs[idx])]
        answer_end_token = input_ids[int(self.end_indexs[idx])]
        answer_start_token = int(
            self.start_indexs[idx]) + len(input_ids_question)
        answer_end_token = int(self.end_indexs[idx]) + len(input_ids_question)
        return input_ids, token_type_ids, attention_mask, answer_start_token, answer_end_token


train_set = QA_Dataset(train_questions_tokenized, train_sentences_tokenized, train_start, train_end)
test_set = QA_Dataset(test_questions_tokenized, test_sentences_tokenized, test_start, test_end)

train_batch_size = 3

train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

"""# Train"""

num_epoch = 30
validation = True
learning_rate = 1.5e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)

total_steps = num_epoch * len(train_loader)
print(total_steps)

model.train()

print("Start Training ...")

best_acc = 0
train_acc = 0
step = 1
for epoch in range(num_epoch):

    train_loss = train_acc = 0

    for data in tqdm(train_loader):
        # Load all data into GPU
        data = [i.to(device) for i in data]

        # Model inputs: input_ids, token_type_ids, attention_mask, start_positions, end_positions (Note: only "input_ids" is mandatory)
        # Model outputs: start_logits, end_logits, loss (return when start_positions/end_positions are provided)
        output = model(input_ids=data[0], token_type_ids=data[1],
                       attention_mask=data[2], start_positions=data[3], end_positions=data[4])

        # Choose the most probable start position / end position
        start_index = torch.argmax(output[1], dim=1)
        end_index = torch.argmax(output[2], dim=1)

        # Prediction is correct only if both start_index and end_index are correct
        train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
        train_loss += output[0]

        output[0].backward()

        optimizer.step()
        optimizer.zero_grad()
        step += 1

    print(f"Epoch {epoch + 1} | Step {step} | loss = {train_loss.item() / len(train_loader):.3f}, acc = {train_acc / len(train_loader):.3f}")

    if validation:
        print("Evaluating Dev Set ...")
        model.eval()
        with torch.no_grad():
            test_acc = 0
            for i, data in enumerate(tqdm(test_loader)):
                data = [i.to(device) for i in data]
                output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2])

                start_index = torch.argmax(output[0], dim=1)
                end_index = torch.argmax(output[1], dim=1)
                answer = tokenizer.decode(data[0][0, start_index: end_index+1])
                test_acc += ((start_index == data[3]) &
                             (end_index == data[4])).float().mean()
            print(
                f"Validation | Epoch {epoch + 1} | acc = {test_acc / len(test_loader):.3f}")
        model.train()

    # if test_acc > best_acc:
    print("Saving Model ...")
    # torch.save(optimizer.state_dict(), './optimizer.pt')
    model_save_dir = "saved_model"
    model.save_pretrained(model_save_dir)
    # best_acc = test_acc