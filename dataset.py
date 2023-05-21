import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import *

transforms = Compose(
    [
        RandomResizedCrop(size=256, scale=(0.6, 1.0)),
        RandomHorizontalFlip(p=0.5),
        RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.95, 1.05)),
        ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4)),
        GaussianBlur(kernel_size=3, sigma=(0.1, 3.0)),
        Resize(size=224),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class CheXpert(Dataset):
    """ Dataset Class"""
    def __init__(
        self,
        img_dir: Union[str, Path] = "ConVIRT",
        csv_dir: Union[str, Path] = "CheXpert-v1.0-small",
        csv_file: str = None, # path to training or validation csv file
        transform = None,
    ):
        self.img_dir = Path(img_dir)
        self.csv_dir = Path(csv_dir)
        self.transform = transform
        self.images, self.labels = self.load_CheXpert(
            img_dir, csv_dir, csv_file, kLoad=True
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None: # transforms done here for half-delay
            image = self.transform(image)

        return image, label

    def load_CheXpert(
        self,
        img_dir: Union[str, Path] = "ConVIRT",
        csv_dir: Union[str, Path] = "CheXpert-v1.0-small",
        csv_file: Union[str, Path] = None,
        # dataframe: str = "train.csv",
        kLoad: bool = False,  # load images or just filenames
    ) -> tuple:
        """

        Parameters
        ----------
        img_dir: Union[str, Path] : (Default value = "CheXpert-v1.0-small")
        csv_dir: Union[str, dataframe]:(Default value = "final_train.csv")
        kLoad :(Default value = False)    

        Returns
        -------
        img: List
        label: List

        """
        dataframe = pd.read_csv(Path(csv_dir) / csv_file)

        images, labels = [], []

        for idx in range(len(dataframe)):
            imgpath = dataframe.loc[idx, "Path"]  # image filename
            label = dataframe.loc[idx, "Report"]  # Get text label

            if kLoad:
                with Image.open(imgpath) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                # if self.transform is not None:
                #     img = self.transform(img)
            else:
                img = imgpath

            images.append(img)
            labels.append(label)

        return images, labels

# import time
# start = time.process_time()
if __name__ == "__main__":
    train_dir = "CheXpert-v1.0-small"
    validation_dir = "CheXpert-v1.0-small"
    training_data = "train.csv"
    validation_data = "final_valid.csv"

    transforms = transforms

    train_dataset = CheXpert(
        img_dir="ConVIRT",
        csv_dir=train_dir,
        csv_file=training_data,
        transform=transforms,
    )

    validation_dataset = CheXpert(
        img_dir="ConVIRT",
        csv_dir=validation_dir,
        csv_file=validation_data,
        transform=transforms,
    )

    print(f"{len(train_dataset)=}")
    print(f"{len(validation_dataset)=}")
# end = time.process_time()
# print("Time elapsed: {} seconds".format(end-start))
