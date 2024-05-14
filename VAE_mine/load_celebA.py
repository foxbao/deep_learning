import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CelebADataset(Dataset):

    def __init__(self, root:str, img_shape=(64, 64)) -> None:
        super().__init__()
        self.root = root
        self.img_shape = img_shape
        self.filenames = sorted(os.listdir(root))

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path).convert('RGB')
        pipeline = transforms.Compose([
            # transforms.CenterCrop(168),
            transforms.Resize(self.img_shape),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])
        return pipeline(img)


def get_dataloader(root='data/celebA/img_align_celeba', batch_size=100,img_shape=(128,128),**kwargs):
    dataset = CelebADataset(root=root,img_shape=img_shape, **kwargs)
    return DataLoader(dataset, batch_size, shuffle=True)


if __name__ == '__main__':
    dataloader = get_dataloader()
    img = next(iter(dataloader))
    print(img.shape)
    # Concat 4x4 images
    N, C, H, W = img.shape
    assert N == 16
    img = torch.permute(img, (1, 0, 2, 3))
    img = torch.reshape(img, (C, 4, 4 * H, W))
    img = torch.permute(img, (0, 2, 1, 3))
    img = torch.reshape(img, (C, 4 * H, 4 * W))
    img = transforms.ToPILImage()(img)
    img.save('work_dirs/tmp.jpg')