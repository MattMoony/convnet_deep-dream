import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def norm(b):
    mean    = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(3, 1, 1)
    std     = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(3, 1, 1)
    return (b - mean) / std

def load_img(p):
    img = Image.open(p)
    img = from_img(img)
    return img

def from_img(img):
    im_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    t = im_transform(img).unsqueeze(0).cuda()
    return t

def to_img(t):
    img = t.cpu().clone().detach()
    img = img.numpy().squeeze()
    img = img.transpose(1, 2, 0)
    img = img.clip(0, 1)

    return img

def zoom(t, scale):
    img = to_img(t)
    h, w, c = img.shape
    img = Image.fromarray((img * 255.).astype(np.uint8))
    res_trans = transforms.Resize((round(h/scale), round(w/scale)))
    img = res_trans(img)
    t = from_img(img)

    return t

def unzoom(t, tsize):
    img = to_img(t)
    h, w, c = img.shape
    img = Image.fromarray((img * 255.).astype(np.uint8))
    res_trans = transforms.Resize(tsize)
    img = res_trans(img)
    t = from_img(img)

    return t

def clip(img):
    mean    = [0.485, 0.456, 0.406]
    std     = [0.229, 0.224, 0.225]

    for c in range(3):
        m, s = mean[c], std[c]
        img[0, c] = torch.clamp(img[0, c], -m / s, (1 - m) / s)

    return img

def save_img(t, p):
    img = to_img(t)
    img = Image.fromarray((img * 255.).astype(np.uint8))
    img.save(p)