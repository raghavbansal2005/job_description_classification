import json
import os
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()



class JdDataSet(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



if __name__ == "__main__":
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    def read_jd_split(file_path, path_to_cat_to_id, texts_column_name, lables_column_name):
        with open(file_path, 'r') as f:
            in_data = json.load(f)
        with open(path_to_cat_to_id, 'r') as f:
            cat_to_id = json.load(f)
        texts = []
        labels = []
        for dp in in_data:
            texts.append(dp[texts_column_name])
            labels.append(cat_to_id[dp[lables_column_name]])

        return texts, labels

    train_texts, train_labels = read_jd_split("./splits/train.json", "./category_to_id.json","jd_sentence_text","jd_sent_manual_label_NEW")
    test_texts, test_labels = read_jd_split("./splits/test.json", "./category_to_id.json","jd_sentence_text","jd_sent_manual_label_NEW")

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = JdDataSet(train_encodings, train_labels)
    val_dataset = JdDataSet(val_encodings, val_labels)
    test_dataset = JdDataSet(test_encodings, test_labels)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    with open(f"./category_to_id.json", 'r') as f:
            cat_to_id = json.load(f)

    num_lables = len(cat_to_id)
    print(num_lables)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = num_lables)
    model.to(device)
    model.train()

    print_gpu_utilization()

    print(len(val_dataset),len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    running_loss = 0

    last_acc = 0
    misc = 0
    total_loss = 0
    final_save_location = 0
    for idx, batch in enumerate(cycle(train_loader)):
        if idx % 15 == 0:
            print(total_loss)
            total_loss = 0
            total_correct = 0
            torch.no_grad()
            model.eval()

            print("Entereing Val")
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                logits = outputs[1]
                n_logits = logits.detach().cpu().numpy()
                max_indexes = np.argmax(n_logits, axis=1)
                total_correct += sum(np.equal(max_indexes, labels.detach().cpu().numpy()) * 1)
                
            current_accu = total_correct/len(val_dataset)
            print("Current group Accuracy:", current_accu)
            if last_acc > current_accu:
                if misc:
                    break
                else:
                    misc = 1
            else:
                misc = 0
                last_acc = current_accu
                final_save_location = idx
                torch.save(model.state_dict(), f"./model_saves/model_at_batch_{idx}.pt")
        else:
            model.train()
            torch.enable_grad()
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            total_loss += loss
            logits = outputs[1]
            running_loss += loss
            logits = outputs[1]
            loss.backward()
            optim.step()

    final_total_correct = 0
    model.load_state_dict(torch.load(f"./model_saves/model_at_batch_{final_save_location}.pt"))
    model.eval()

    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        n_logits = logits.detach().cpu().numpy()
        max_indexes = np.argmax(n_logits, axis=1)
        final_total_correct += sum(np.equal(max_indexes, labels.detach().cpu().numpy()) * 1)
    
    print("Accuracy: ", final_total_correct/len(test_dataset))

    # for epoch in range(5):
    #     model.train()
    #     torch.enable_grad()
    #     print(f"Epoch {epoch} is starting")
    #     for batch in train_loader:
    #         optim.zero_grad()
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         labels = batch['labels'].to(device)
    #         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    #         loss = outputs[0]
    #         logits = outputs[1]
    #         running_loss += loss
    #         logits = outputs[1]
    #         loss.backward()
    #         optim.step()
            
        
    #     print("Total Loss: ", running_loss)   
    #     running_loss = 0

    #     total_correct = 0
    #     torch.no_grad()
    #     model.eval()


    #     for batch in val_loader:
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         labels = batch['labels'].to(device)
    #         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    #         loss = outputs[0]
    #         logits = outputs[1]
    #         n_logits = logits.detach().numpy()
    #         max_indexes = np.argmax(n_logits, axis=1)
    #         total_correct += sum(np.equal(max_indexes, labels) * 1)
        
    #     print("Accuracy: ", total_correct/len(val_dataset))


    # final_total_correct = 0
    # for batch in test_loader:
    #     input_ids = batch['input_ids'].to(device)
    #     attention_mask = batch['attention_mask'].to(device)
    #     labels = batch['labels'].to(device)
    #     outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    #     loss = outputs[0]
    #     logits = outputs[1]
    #     n_logits = logits.detach().numpy()
    #     max_indexes = np.argmax(n_logits, axis=1)
    #     final_total_correct += sum(np.equal(max_indexes, labels) * 1)
    
    # print("Accuracy: ", final_total_correct/len(test_dataset))

    
