from typing import List

import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, AutoTokenizer, BertModel


class text_encoder(nn.Module):
    """Text encoder for generating text representations"""

    def __init__(self):
        super(text_encoder, self).__init__()

        self.layer_1 = nn.Linear(in_features=768, out_features=768)
        self.act = nn.ReLU()
        self.layer_2 = nn.Linear(in_features=768, out_features=512)

    def max_pooling(self, model_output, attention_mask):
        """

        Parameters
        ----------
        model_output : ndarray

        attention_mask : ndarray


        Returns
        -------
        embeddings : ndarray

        """
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        token_embeddings[input_mask_expanded == 0] = -1e9

        return torch.max(token_embeddings, 1)[0]

    def forward(self, text: List):
        """

        Parameters
        ----------
        text: List :


        Returns
        -------
        text_vector : ndarray

        """
        model = AutoModel.from_pretrained("bert_model")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        for param in model.embeddings.parameters():
            param.requires_grad = False

        for param in model.encoder.layer[0:6].parameters():
            param.requires_grad = False

        encoded_input = tokenizer.batch_encode_plus(
            text, padding=True, truncation=True, return_tensors="pt"
        )

        model_output = model(**encoded_input)

        out_1 = self.act(
            self.layer_1(
                self.max_pooling(model_output, encoded_input["attention_mask"])
            )
        )

        text_vector = self.layer_2(out_1)

        return text_vector
