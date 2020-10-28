import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
import torch.utils.data as Data
import torch

def pil_loader(path):  # 一般采用pil_loader函数。
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


# dataset = dset.ImageFolder('DBtest/img/data/test', loader=pil_loader, transform=transforms.ToTensor())
# train_loader = Data.DataLoader(dataset=dataset, batch_size=len(dataset.samples), shuffle=True)
# for i, (imgs, targets) in enumerate(train_loader):
#     text_x = imgs
#     text_y = targets
# text_y = text_y.numpy()
use_gpu = torch.cuda.is_available()
