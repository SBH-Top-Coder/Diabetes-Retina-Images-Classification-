import warnings
warnings.filterwarnings("ignore")
import Config
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


class DRDataset(Dataset):
    def __init__(self, images_folder, path_to_csv, transform=None):
        super().__init__()
        self.data = pd.read_csv(path_to_csv)
        self.images_folder = images_folder
        self.image_files = os.listdir(images_folder)
        self.transform = transform
        # self.train = train

    def __len__(self):
        return self.data.shape[0] 

    def __getitem__(self, index):
        image_file, label = self.data.iloc[index]
        image = np.array(Image.open(os.path.join(self.images_folder, image_file+".jpeg")))
        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label, image_file

# if __name__ == "__main__":
#     """
#     Test if everything works ok
#     """
#     dataset = DRDataset(
#         images_folder="/home/semi/Image_Classification/Full_Preproceed/",
#         path_to_csv="/home/semi/Image_Classification/Labels.csv",
#         transform=Config.val_transforms,
#     )
#     loader = DataLoader(
#         dataset=dataset, batch_size=16, num_workers=2, shuffle=True, pin_memory=True
#     )

#     for x, label, file in tqdm(loader):
#         print(x.shape)
#         print(label.shape)
#         import sys
#         sys.exit()