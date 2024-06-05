
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import os
import torchvision.transforms as transforms
class ParkFullDataset(Dataset):
    def __init__(self,img_dir,img_names,layout_dir,layout_names,text_dir,text_names,img_shape=(128, 128),layout_shape=(32,32),use_layout=False,text_context=False) -> None:
        super().__init__()
        self.use_layout = use_layout
        self.text_context=text_context
        self.img_dir =img_dir 
        df = pd.read_csv(img_names, header=None,delimiter='\t', names=["Filename"])
        self.img_names= df["Filename"].values
        
        self.layout_dir=layout_dir
        df2 = pd.read_csv(layout_names, header=None,delimiter='\t', names=["Filename"])
        self.layout_names=df2["Filename"].values
        
        self.text_dir=text_dir
        df3 = pd.read_csv(text_names, header=None,delimiter='\t', names=["Filename"])
        self.text_names=df3["Filename"].values
        self.img_transform=transforms.Compose([
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.transform_layout=transforms.Compose([
            transforms.Resize(layout_shape),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    # Return the number of images in the dataset
    def __len__(self):
        return len(self.img_names)
    
    # Get the image and label at a given index
    def __getitem__(self, idx):
        # Return the image and label as a tuple
        image = self.img_transform(Image.open(os.path.join(self.img_dir, self.img_names[idx])))
            # plt.imshow(image)
        if self.use_layout:
            layout = self.transform_layout(Image.open(os.path.join(self.layout_dir, self.layout_names[idx])))
        else:
            layout=torch.tensor(0).to(torch.int64)
        
        if self.text_context:
            with open(os.path.join(self.text_dir, self.text_names[idx]), "r", encoding="utf-8") as file:
                text = file.read()
        else:
            text=""
        
        return (image,layout,text)

    def getshapes(self):
        # return shapes of data and labels
        return self.sprites_shape, self.slabel_shape