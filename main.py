import json
import os
import numpy as np
import torch
from tokenizers import BertWordPieceTokenizer
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
# import nltk
# from nltk.corpus import stopwords
from models import LSTM_model, InnerProduct


BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 0.65
EPOCHS = 13

class Articles(torch.utils.data.Dataset):
    def __init__(self, json_file):
        super().__init__()
        with open(json_file, "r") as data_file:
            self.examples = json.loads(data_file.read())

    def __getitem__(self,idx):
        return self.examples[idx]
    
    def __len__(self):
        return len(self.examples)


    def map_items(
        self,
        tokenizer,
        category_to_id
    ):
        for idx, example in enumerate(self.examples):
            encoded = tokenizer.encode(example["jd_sentence_text"], add_special_tokens=False).ids
            self.examples[idx]["mapped_sentence"] = encoded
            self.examples[idx]["label"] = category_to_id.get(example["jd_sent_manual_label_NEW"])

def collate_fn_lstm(examples):
    words = []
    labels = []
    for example in examples:
        words.append(example['jd_sentence_text'])
        labels.append(example['label'])

    batch_words = [tokenizer.encode(word) for word in words]
    num_words = [len(x) for x in batch_words]
    [entry.pad(max(num_words)) for entry in batch_words]
    batch_words = [entry.ids for entry in batch_words]
    batch_words = torch.tensor(batch_words, dtype=torch.float32)
    lengths = torch.tensor(num_words, dtype=torch.float32)

    labels_arr = torch.zeros((len(examples), len(category_to_id)), dtype=torch.float32)
    for i in range(len(labels)):
        labels_arr[i][labels[i]] = 1
    return batch_words, lengths, labels_arr

def collate_fn_RFS(examples):
    words = []
    articles = []
    labels = []
    # publications = []
    for example in examples:
        if True:
            words.append(example['jd_sentence_text'])
        else:
            if len(example['text']) > args.words_to_use:
                words.append(list(set(example['text'][:args.words_to_use])))
            else:
                words.append(list(set(example['text'])))
        articles.append(1)
        labels.append(example['label'])

    #print(words)
    batch_words = [tokenizer.encode(word) for word in words]
    num_words = [len(x) for x in batch_words]
    batch_words = [entry.ids for entry in batch_words]
    # print(batch_words)
    words = np.concatenate(batch_words, axis=0)
    word_attributes = torch.tensor(words, dtype=torch.long)
    articles = torch.tensor(articles, dtype=torch.long)
    num_words.insert(0, 0)
    num_words.pop(-1)
    attribute_offsets = torch.tensor(np.cumsum(num_words), dtype=torch.long)
    # publications = torch.tensor(publications, dtype=torch.long)
    labels_arr = torch.zeros((len(examples), len(category_to_id)), dtype=torch.float32)
    for i in range(len(labels)):
        labels_arr[i][labels[i]] = 1
    
    publications = torch.zeros(len(labels),dtype=torch.long)
    for i in range(len(labels)):
        publications[i] = labels[i]
    #real_labels = torch.tensor(labels, dtype=torch.long)
    return articles, word_attributes, attribute_offsets, labels_arr, publications


if __name__ == "__main__":

    train_data = Articles("splits/train.json")
    test_data = Articles("splits/test.json")
    val_data = Articles("splits/val.json")

    with open("category_to_id.json", 'r') as f:
        category_to_id = json.load(f)

    tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
    num_words = tokenizer.get_vocab_size()

    train_data.map_items(tokenizer, category_to_id)
    test_data.map_items(tokenizer, category_to_id)
    val_data.map_items(tokenizer, category_to_id)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, collate_fn=collate_fn_RFS)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=EVAL_BATCH_SIZE, drop_last=True, shuffle=True, collate_fn=collate_fn_RFS)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=EVAL_BATCH_SIZE, drop_last=True, shuffle=True, collate_fn=collate_fn_RFS)

    model = InnerProduct(len(category_to_id),1,num_words, 512,False,False,"mean")

    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)
    running_loss = 0

    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch}")
        model.train()
        torch.enable_grad()
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            articles, batch_words, lengths, labels, publications = batch    # articles, word_attributes, attribute_offsets, real_labels
            logits = model(publications,articles,batch_words,lengths)   # publications, articles, word_attributes, attribute_offsets
            L = loss(logits, labels)
            L.backward()
            optimizer.step()
            running_loss += L.item()
            
        print("Total Loss: ", running_loss)
        running_loss = 0

        total_correct = 0
        torch.no_grad()
        model.eval()

        for idx, batch in enumerate(val_loader):
            articles, batch_words, lengths, labels, publications = batch
            logits = model(publications,articles,batch_words,lengths)
            n_logits = logits.detach().numpy()
            n_labels = labels.numpy()
            max_indexes = np.argmax(n_logits, axis=1)
            actual_labels = np.argmax(n_labels, axis=1)
            total_correct += sum(np.equal(max_indexes, actual_labels) * 1)

        print("Accuracy: ", (total_correct/len(val_data)))   


    
    total_correct = 0
    for idx, batch in enumerate(test_loader):
        articles, batch_words, lengths, labels, publications = batch
        logits = model(publications,articles,batch_words,lengths)
        n_logits = logits.detach().numpy()
        n_labels = labels.numpy()
        max_indexes = np.argmax(n_logits, axis=1)
        actual_labels = np.argmax(n_labels, axis=1)
        total_correct += sum(np.equal(max_indexes, actual_labels) * 1)

    print("Final Accuracy: ", (total_correct/len(test_data)))
            


        


