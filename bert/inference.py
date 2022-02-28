import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForQuestionAnswering.from_pretrained("./bert/saved_model").to(device)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def splitSpeech(speech):
    model.eval()
    with torch.no_grad():
        question = 'What should I get for her?'
        sentence = speech
        question_tokenized = tokenizer(question, add_special_tokens=False)
        sentence_tokenized = tokenizer(sentence, add_special_tokens=False)

        input_ids_question = [101] + question_tokenized.input_ids + [102]
        input_ids_sentence = sentence_tokenized.input_ids + [102]
        input_ids = torch.tensor(input_ids_question + input_ids_sentence).unsqueeze(0)

        token_type_ids = torch.tensor(
            [0] * len(input_ids_question) + [1] * len(input_ids_sentence)).unsqueeze(0)

        attention_mask = torch.tensor([1] * len(input_ids[0])).unsqueeze(0)

        model.eval()

        output = model(input_ids=input_ids.to(device), token_type_ids=token_type_ids.to(
            device), attention_mask=attention_mask.to(device))

        # Choose the most probable start position / end position
        start_index = torch.argmax(output[0], dim=1)
        end_index = torch.argmax(output[1], dim=1)

        # part_object = tokenizer.decode(input_ids[0, start_index: end_index+1])
        # part_action = tokenizer.decode(input_ids[0, len(input_ids_question):start_index])
        part_action = input_ids[0, len(input_ids_question):start_index].tolist()
        part_action = [101] + part_action + [102]
        
        part_object = input_ids[0, start_index: end_index+1].tolist()
        part_object = [101] + part_object + [102]
        
        return part_action, part_object


if __name__ == '__main__':
    command = 'Please give me the medicine jar with blue cover'
    part_action, part_object = splitSpeech(command)
    print(part_action, ' \ ', part_object)
