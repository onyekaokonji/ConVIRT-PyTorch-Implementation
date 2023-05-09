import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

from image_encoder import image_encoder
from text_encoder import text_encoder


class total_loss(nn.Module):
    def __init__(
        self,
        batch_size: int,  # batch size for training
        loss_weight: int,  # loss weight for training
        temperature: int,  # temperature for training
        image_vector: int,  # image vector for training
        text_vector: int,  # text vector for training
    ):
        self.batch_size = batch_size
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.image_vector = image_vector
        self.text_vector = text_vector

    def forward(self):
        image_to_text_cosine_similarity_score = cosine_similarity(
            self.image_vector, self.text_vector
        )

        text_to_image_cosine_similarity_score = cosine_similarity(
            self.text_vector, self.image_vector
        )

        image_to_text_loss = -torch.log(
            (torch.exp(image_to_text_cosine_similarity_score / self.temperature))
            / torch.sum(
                torch.exp(image_to_text_cosine_similarity_score / self.temperature)
            )
        )

        text_to_image_loss = -torch.log(
            (torch.exp(text_to_image_cosine_similarity_score / self.temperature))
            / torch.sum(
                torch.exp(text_to_image_cosine_similarity_score / self.temperature)
            )
        )

        total_loss = (1 / self.batch_size) * torch.sum(
            self.loss_weight * image_to_text_loss
            + (1 - self.loss_weight) * text_to_image_loss
        )

        return total_loss


# def total_loss(batch_size, loss_weight, temperature, image_vector, text_vector):

#   image_to_text_cosine_similarity_score = cosine_similarity(image_vector, text_vector)

#   text_to_image_cosine_similarity_score = cosine_similarity(text_vector, image_vector)

#   image_to_text_loss = -np.log((np.exp(image_to_text_cosine_similarity_score / temperature)) /
#                           np.sum(np.exp(image_to_text_cosine_similarity_score / temperature)))

#   text_to_image_loss = -np.log((np.exp(text_to_image_cosine_similarity_score / temperature)) /
#                             np.sum(np.exp(text_to_image_cosine_similarity_score / temperature)))

#   total_loss = (1/batch_size) * np.sum(loss_weight * image_to_text_loss + (1 - loss_weight) * text_to_image_loss)

#   return total_loss
