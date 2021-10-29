from model import VAE
import torch
import numpy as np
import torchvision
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import torch.nn.functional as F
from glob import glob
import pandas
#from google.colab.patches import cv2_imshow
import utils
import os
import argparse

device = 'cuda:0'


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    #about execute option
    parser.add_argument('--dataset', type=str, default='toothbrush', choices=['toothbrush', 'bottle', 'capsule'])
    parser.add_argument('--dataroot', type=str, default='datasets/')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

    #about training
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)

    #about test option
    parser.add_argument('--weight_path', type=str, default='datasets/toothbrush/model.pth')

    #about save
    parser.add_argument('--save_img', type=str, default='result/img')
    parser.add_argument('--save_pth', type=str, default='result/path')

    return parser


class MVDataset(Dataset):
    def __init__(self, config):
        self.root = os.path.join(config.dataroot, config.dataset)
        self.mode = config.mode
        self.x_data = []
        self.y_data = []

        if self.mode == 'train':
            self.root = os.path.join(self.root, self.mode, 'good/')
            self.img_path = sorted(glob(self.root + '*.png'))
 
        elif self.mode == 'test':
            self.root = os.path.join(self.root, self.mode, 'defective/')
            self.img_path = sorted(glob(self.root + '*.png'))

        for i in tqdm.tqdm(range(len(self.img_path))):
            img = cv2.imread(self.img_path[i], cv2.IMREAD_COLOR)
            print(self.img_path[i])
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            img = cv2.resize(img, dsize=(256, 256))
            #cv2.imwrite('test_%d.png' % i, img)

            self.x_data.append(img)
            self.y_data.append(img)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        transform1 = torchvision.transforms.ToTensor()
        new_x_data = transform1(self.x_data[idx])
        return new_x_data, self.y_data[idx]


class Trainer(object):
    def __init__(self, config):
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.lr
        self._build_model()
        self.binary_cross_entropy = torch.nn.BCELoss()

        dataset = MVDataset(config)
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate, betas=(0.9, 0.999)) #todo optimization

        # Load of pretrained_weight file
        print("Training...")

    def _build_model(self):
        net = VAE()
        self.net = net.to(device)
        self.net.train()

        print('Finish build model.')

    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = self.binary_cross_entropy(recon_x.view(-1, 256*256*3), x.view(-1, 256*256*3))
        kldivergence = -0.5 * torch.sum(1+ logvar - mu.pow(2) - logvar.exp()) 
        return recon_loss + 0.000001 * kldivergence

    def train(self, config):
        path = os.path.join(config.save_pth, config.dataset)
        if not os.path.exists(path):
            os.makedirs(path)
            print('make directory ...', path)

        for epoch in tqdm.tqdm(range(self.epochs + 1)):
            if epoch % 200 == 0:
                torch.save(self.net.state_dict(), "_".join(['model', str(epoch), '.pth'])) #Change this path


            for batch_idx, samples in enumerate(self.dataloader):
                x_train, y_train = samples
                x_train, y_train = x_train.to(device), y_train.to(device)

                g, latent_mu, latent_var = self.net.forward(x_train)
                loss = self.vae_loss(g, x_train, latent_mu, latent_var)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
             

        print('Finish training.')


class Tester(object):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self._build_model()

        dataset = MVDataset(config, method='test')
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.datalen = dataset.__len__()
        self.mse_all_img = []

        # Load of pretrained_weight file
        weight_PATH = config.weight_path
        self.net.load_state_dict(torch.load(weight_PATH))

        print("Testing...")

    def _build_model(self):
        net = VAE()
        self.net = net.to(device)
        self.net.train()

        print('Finish build model.')

    def test(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            print('make dir path:', save_path)

        for batch_idx, samples in enumerate(self.dataloader):
            x_test, y_test = samples
            out = self.net(x_test.cuda())

            x_test2 = 256. * x_test
            out2 = 256. * out[0]

            abnomal = utils.compare_images_colab(x_test2[0].clone().permute(1, 2, 0).cpu().detach().numpy(), out2[0].clone().permute(1, 2, 0).cpu().detach().numpy(), None, 0.2)

            ori_path = os.path.join(save_path, 'test_%d_ori.png' % batch_idx)
            gen_path = os.path.join(save_path, 'test_%d_gen.png' % batch_idx)
            diff_path = os.path.join(save_path, 'test_%d_diff.png' % batch_idx)

            cv2.imwrite(ori_path, cv2.cvtColor(x_test2[0].clone().permute(1, 2, 0).cpu().detach().numpy(), cv2.COLOR_RGB2BGR))
            cv2.imwrite(gen_path, cv2.cvtColor(out2[0].clone().permute(1, 2, 0).cpu().detach().numpy(), cv2.COLOR_RGB2BGR))
            cv2.imwrite(diff_path, abnomal)



def main():

    config = get_command_line_parser().parse_args()

    trainer = Trainer(config)
    trainer.train(config)

    #tester = Tester(config)
    #tester.test(config.save_pth)

if __name__ == '__main__':
    main()
