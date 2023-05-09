import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset import *
from image_encoder import image_encoder
from losses import total_loss
from text_encoder import text_encoder


class Model:
    """Model class for training"""

    def __init__(
        self,
        training_dataframe: str,
        validation_dataframe: str,
        train_image_dir: str,
        validation_image_dir: str,
        config: str,
    ):
        self.training_dataframe = training_dataframe
        self.validation_dataframe = validation_dataframe
        self.train_image_dir = train_image_dir
        self.validation_image_dir = validation_image_dir
        self.config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def core(self):
        """Core training method"""
        i_e = image_encoder()
        t_e = text_encoder()
        train_dataset = CheXpert(
            self.train_image_dir, self.training_dataframe, transform=transforms
        )

        validation_dataset = CheXpert(
            self.validation_image_dir, self.validation_dataframe, transform=transforms
        )

        train_ds = DataLoader(train_dataset, batch_size=16)

        valid_ds = DataLoader(validation_dataset, batch_size=16)

        train_loss = 0.0
        validation_loss = 0.0

        for epoch in range(1, self.config["n_epochs"] + 1):
            for image, text in train_ds:
                image = image.to(self.device)
                text = text.to(self.device)

                image_vector = i_e(image)
                text_vector = t_e(list(text))

                optimizer = torch.optim.Adam(i_e.parameters())

                optimizer.zero_grad()

                loss = total_loss(
                    batch_size=self.config["batch_size"],
                    loss_weight=self.config["loss_weight"],
                    temperature=self.config["temperature"],
                    image_vector=image_vector,
                    text_vector=text_vector,
                )

                loss.backward()

                optimizer.step()

                with torch.no_grad():
                    i_e.eval()
                    t_e.eval()

                    for valid_image, valid_text in valid_ds:
                        valid_image = valid_image.to(self.device)
                        valid_text = valid_text.to(self.device)

                        valid_image_vector = i_e(valid_image)
                        valid_text_vector = t_e(list(valid_text))

                        vloss = total_loss(
                            batch_size=self.config["batch_size"],
                            loss_weight=self.config["loss_weight"],
                            temperature=self.config["temperature"],
                            image_vector=valid_image_vector,
                            text_vector=valid_text_vector,
                        )

                        vloss.backward()

                        validation_loss += vloss.item()

                train_loss += loss.item()

                print(f"Epoch: {epoch} Training Loss: {train_loss/len(train_ds)}")
                print(f"Epoch: {epoch} Validation Loss: {train_loss/len(valid_ds)}")


if __name__ == "__main__":
    model = Model(
        "final_train.csv",
        "final_valid.csv",
        "CheXpert-v1.0-small/train",
        "CheXpert-v1.0-small/valid",
        "config.yaml",
    )
    model.core()
