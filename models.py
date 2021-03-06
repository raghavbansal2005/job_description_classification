from numpy import inner
import torch


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTM_model(nn.Module):
    def __init__(self, num_words, emb_size, outputs, dropout):
        super().__init__()
        self.word_embeddings = nn.Embedding(num_words, emb_size)
        self.lstm = nn.LSTM(emb_size, emb_size, batch_first = True)
        self.linear = nn.Linear(emb_size, outputs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):

        x = self.word_embeddings(x)
        x_pack = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        outputs, (hn, cn) = self.lstm(x_pack)
        dense_outputs=self.linear(hn[-1])

        #outputs = nn.Softmax(dense_outputs)
        return dense_outputs


class InnerProduct(nn.Module):
    def __init__(self, n_publications, n_articles, n_attributes, emb_size, sparse, use_article_emb, mode):
        super().__init__()
        self.emb_size = emb_size
        self.publication_embeddings = nn.Embedding(n_publications, emb_size, sparse=sparse)
        self.publication_bias = nn.Embedding(n_publications, 1, sparse=sparse)
        self.attribute_emb_sum = nn.EmbeddingBag(n_attributes, emb_size, mode=mode, sparse=sparse)
        self.attribute_bias_sum = nn.EmbeddingBag(n_attributes, 1, mode=mode, sparse=sparse)
        self.use_article_emb = use_article_emb
        self.n_publications = n_publications
        if use_article_emb:
            self.article_embeddings = nn.Embedding(n_articles, emb_size, sparse=sparse)
            self.article_bias = nn.Embedding(n_articles, 1, sparse=sparse)
        self.use_article_emb = use_article_emb

    def reset_parameters(self):
        for module in [self.publication_embeddings, self.attribute_emb_sum]:
            scale = 0.07
            nn.init.uniform_(module.weight, -scale, scale)
        for module in [self.publication_bias, self.attribute_bias_sum]:
            nn.init.zeros_(module.weight)
        if self.use_article_emb:
            for module in [self.article_embeddings, self.article_bias]:
                # initializing article embeddings to zero to allow large batch sizes
                # nn.init.uniform_(module.weight, -scale, scale)
                nn.init.zeros_(module.weight)

    def forward(self, publications, articles, word_attributes, attribute_offsets, pairwise=True, return_intermediate=False):
        publication_emb = self.publication_embeddings(publications)
        attribute_emb = self.attribute_emb_sum(word_attributes, attribute_offsets)
        if self.use_article_emb:
            article_and_attr_emb = self.article_embeddings(articles) + attribute_emb
        else:
            article_and_attr_emb = attribute_emb
        attr_bias = self.attribute_bias_sum(word_attributes, attribute_offsets)
        publication_bias = self.publication_bias(publications)
        if pairwise:
            # for every publication, compute inner product with every article
            # (publications, emb_size) x (emb_size, articles) -> (publications, articles)
            inner_prod = publication_emb @ article_and_attr_emb.t()
            # broadcasting across publication dimension
            logits = inner_prod + publication_bias
            # broadcast across article dimension
            logits += attr_bias.t()
            if self.use_article_emb:
                logits += self.article_bias(articles).t()
            logits = torch.transpose(logits[:self.n_publications],0,1)
        else:
            # for every publication, only compute inner product with corresponding minibatch element
            # (batch_size, 1, emb_size) x (batch_size, emb_size, 1) -> (batch_size, 1)
            # logits = torch.bmm(publication_emb.view(-1, 1, self.emb_size),
            #                    (article_and_attr_emb).view(-1, self.emb_size, 1)).squeeze()
            inner_prod = (publication_emb * article_and_attr_emb).sum(-1)
            logits = inner_prod + attr_bias.squeeze() + publication_bias.squeeze()
            if self.use_article_emb:
                logits += self.article_bias(articles).squeeze()
        if return_intermediate:
            return logits, publication_emb, attribute_emb
        else:
            return logits