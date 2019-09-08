import torch
from torch import optim, nn, autograd
from torchvision import models, transforms

import os
import utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser

class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        lenet = models.googlenet(pretrained=True)

        self.conv1          = lenet.conv1
        self.maxpool1       = lenet.maxpool1
        self.conv2          = lenet.conv2
        self.conv3          = lenet.conv3
        self.maxpool2       = lenet.maxpool2

        self.inception3a    = lenet.inception3a
        self.inception3b    = lenet.inception3b
        self.maxpool3       = lenet.maxpool3

        self.inception4a    = lenet.inception4a
        self.inception4b    = lenet.inception4b
        self.inception4c    = lenet.inception4c
        self.inception4d    = lenet.inception4d
        self.inception4e    = lenet.inception4e
        self.maxpool4       = lenet.maxpool4

        self.inception5a    = lenet.inception5a
        self.inception5b    = lenet.inception5b

        self.avgpool        = lenet.avgpool
        self.dropout        = lenet.dropout

        self.layers = [
            self.conv1, self.maxpool1, self.conv2, self.conv3, self.maxpool2,
            self.inception3a, self.inception3b, self.maxpool3, 
            self.inception4a, self.inception4b, self.inception4c, self.inception4d, self.inception4e, self.maxpool4, 
            self.inception5a, self.inception5b,
            self.avgpool, self.dropout
        ]

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x, layer_n=10):
        h = x
        for i in range(layer_n):
            h = self.layers[i](h)

        return h

def l2_norm(actvs):
    return torch.sqrt(torch.sum(actvs.pow(2)))

def dreamchapter(model, img, lr=0.01, iters=20, verbose=True, interval=5, jitter=32, layer_n=10):
    plt.ion()
    for i in range(iters):
        with torch.no_grad():
            rx, ry = torch.randint(-jitter, jitter+1, (2,))
            img = torch.roll(img, (rx, ry), (-1, -2))

        img.requires_grad_(True)
        actvs = model(utils.norm(img), layer_n)

        lss = l2_norm(actvs)
        lss.backward()

        with torch.no_grad():
            img += lr / torch.abs(img.grad).mean() * img.grad
            img.grad.zero_()
            
            img = utils.clip(img)
            img = torch.roll(img, (-rx, -ry), (-1, -2))

        if verbose and (i % interval == 0):
            plt.imshow(utils.to_img(img))
            plt.title('Partial-Dream[iter#{:04d}]'.format(i))
            plt.pause(1e-3)

    plt.close('all')
    plt.ioff()

    return img

def deepdream(model, imgp, n_octaves=10, octave_scale=1.4, lr=0.01, iters=20, verbose=True, interval=5, layer_n=10):
    model.eval()
    img = utils.load_img(imgp)

    octaves = [img]
    for _ in range(n_octaves - 1):
        octaves.append(utils.zoom(octaves[-1], octave_scale))

    detail = torch.zeros(*octaves[-1].size()).cuda()
    for i, octave in enumerate(octaves[::-1]):
        if i > 0:
            detail = utils.unzoom(detail, octave.size()[2:])

        currn_img = octave + detail
        dream_img = dreamchapter(model, currn_img, lr=lr, iters=iters, verbose=verbose, interval=interval, layer_n=layer_n)
        detail = dream_img - octave

    return dream_img

def main():
    parser = ArgumentParser()

    parser.add_argument('-b', '--base-img', type=str, dest='base_img', required=True, help='Path to the base image ... ')
    parser.add_argument('-d', '--destination', type=str, dest='destination', help='Path for the final image ... ')

    parser.add_argument('-n', '--n-octaves', type=int, dest='n_octaves', default=10, help='Amount of octaves ... ')
    parser.add_argument('-s', '--octave-scale', type=float, dest='octave_scale', default=1.4, help='The octave scaling factor ... ')

    parser.add_argument('--lr', type=float, dest='lr', default=0.01, help='Learning rate / step size ... ')
    parser.add_argument('--iters', type=int, dest='iters', default=10, help='Amount of iterations per octave ... ')
    parser.add_argument('--layer-n', type=int, dest='layer_n', default=10, help='Layer-activation to maximize [1;18] ... ')

    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', default=False, help='Flag; Verbose?')
    parser.add_argument('-i', '--interval', type=int, dest='interval', default=5, 
                        help='Interval for displaying intermediate results ... ')

    args = parser.parse_args()

    if not os.path.isfile(args.base_img):
        print('[-] Base-Image doesn\'t exist ... ({:})'.format(args.base_img))
        os._exit(1)

    if args.destination and (
        os.path.isfile(args.destination) or 
        not os.path.isdir(os.path.dirname(args.destination))
    ):
        print('[-] Final path either already present, or un-reachable ... ({:})'.format(args.destination))
        os._exit(1)

    model = GoogLeNet()
    model = model.cuda()

    dream = deepdream(model, args.base_img, n_octaves=args.n_octaves, octave_scale=args.octave_scale, 
                        lr=args.lr, iters=args.iters, verbose=args.verbose, interval=args.interval, layer_n=args.layer_n)

    plt.imshow(utils.to_img(dream))
    plt.title('Deep-Dream')
    plt.show()

    if not args.destination:
        yN = input('Save image? [y/N] ')
        if yN in ['y', 'Y']:
            path = ''
            while (
                os.path.isfile(path) or 
                not os.path.isdir(os.path.dirname(path))
            ):
                path = input('Enter path: ')

            utils.save_img(dream, path)
    else:
        utils.save_img(dream, args.destination)

if __name__ == '__main__':
    main()