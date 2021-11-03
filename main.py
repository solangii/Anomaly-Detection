import argparse
from glob import glob

import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from model import VAE
from utils import *

device = 'cuda:0'


def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about execute option
    parser.add_argument('--dataset', type=str, default='toothbrush', choices=['toothbrush', 'bottle', 'capsule'])
    parser.add_argument('--dataroot', type=str, default='datasets/')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test/defective', 'test/poke', 'test/squeeze', 'test/broken_large',
                                 'test/broken_small', 'test/contamination', 'test/crack'])
    parser.add_argument('--seed', type=int, default=1)

    # about training
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--constant', type=float, default=1e-6, help='constant for loss')

    # about test option
    parser.add_argument('--weight_path', type=str, default='datasets/toothbrush/model.pth', help='used for test')
    parser.add_argument('--threshold', type=float, default=0.2, help='using at calculating difference')

    # about save
    parser.add_argument('--save_root', type=str, default='result/')
    parser.add_argument('--memo', type=str, default='')

    config = parser.parse_args()

    return config


class MVDataset(Dataset):
    def __init__(self, config):
        self.root = os.path.join(config.dataroot, config.dataset)  # root : datasets/[dataset]
        self.mode = config.mode

        self.x_data = []
        self.y_data = []

        if self.mode == 'train':
            self.root = os.path.join(self.root, self.mode, 'good/')  # root : datasets/[dataset]/train/good/
            self.img_path = sorted(glob(self.root + '*.png'))
        else:
            method, section = self.mode.split('/')
            self.root = os.path.join(self.root, method, section)  # root : datasets/[dataset]/test/[section]
            self.root += '/'
            self.img_path = sorted(glob(self.root + '*.png'))

        for i in tqdm(range(len(self.img_path))):
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
        self._build_model()
        self.binary_cross_entropy = torch.nn.BCELoss()

        dataset = MVDataset(config)
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr, betas=(0.9, 0.999))

        # Load of pretrained_weight file
        print("Training...")

    def _build_model(self):
        net = VAE()
        self.net = net.to(device)
        self.net.train()

        print('Finish build model.')

    def vae_loss(self, recon_x, x, mu, logvar, constant):
        recon_loss = self.binary_cross_entropy(recon_x.view(-1, 256 * 256 * 3), x.view(-1, 256 * 256 * 3))
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + constant * kldivergence

    def train(self, config):
        epochs = config.epochs
        for epoch in tqdm(range(epochs + 1)):
            # if save params when specific epoch, use below
            """
            if epoch == 10 or epoch == 500:
                name = exp_name(epoch, config.lr, config.batch_size, config.constant)
                path = os.path.join(config.save_path, name)

                path += ".pth"
                print("save params to ... ", path)

                torch.save(self.net.state_dict(), path)"""

            for batch_idx, samples in enumerate(self.dataloader):
                x_train, y_train = [_.cuda() for _ in samples]

                g, latent_mu, latent_var = self.net.forward(x_train)
                loss = self.vae_loss(g, x_train, latent_mu, latent_var, config.constant)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        name = exp_name(epochs, config.lr, config.batch_size, config.constant)
        path = os.path.join(config.save_path, name)
        path += ".pth"

        print("save params to ... ", path)
        torch.save(self.net.state_dict(), path)

        print('Finish training.')


class Tester(object):
    def __init__(self, config):
        self._build_model()

        dataset = MVDataset(config)
        self.root = dataset.root
        self.dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        self.datalen = dataset.__len__()
        self.mse_all_img = []

        # Load of pretrained_weight file
        weight_PATH = config.weight_path
        print('using weight ...', weight_PATH)

        self.net.load_state_dict(torch.load(weight_PATH))

        print("Testing...")

    def _build_model(self):
        net = VAE()
        self.net = net.to(device)
        self.net.eval()

        print('Finish build model.')

    def test(self, config):
        save_path = config.save_path
        print('save to ...', save_path)

        if config.memo != '':
            save_path += '/'
            save_path += config.memo

        if not os.path.exists(save_path):
            os.mkdir(save_path)
            print('make dir path:', save_path)

        for batch_idx, samples in enumerate(self.dataloader):
            x_test, y_test = samples
            out = self.net(x_test.cuda())

            x_test2 = 256. * x_test
            out2 = 256. * out[0]

            abnomal = compare_images_colab(x_test2[0].clone().permute(1, 2, 0).cpu().detach().numpy(),
                                           out2[0].clone().permute(1, 2, 0).cpu().detach().numpy(), None,
                                           config.threshold)

            ori_path = os.path.join(save_path, 'test_%d_ori.png' % batch_idx)
            gen_path = os.path.join(save_path, 'test_%d_gen.png' % batch_idx)
            diff_path = os.path.join(save_path, 'test_%d_diff.png' % batch_idx)

            cv2.imwrite(ori_path,
                        cv2.cvtColor(x_test2[0].clone().permute(1, 2, 0).cpu().detach().numpy(), cv2.COLOR_RGB2BGR))
            cv2.imwrite(gen_path,
                        cv2.cvtColor(out2[0].clone().permute(1, 2, 0).cpu().detach().numpy(), cv2.COLOR_RGB2BGR))
            cv2.imwrite(diff_path, abnomal)

        print('Finish test')


def main():
    config = get_command_line_parser()
    set_seed(config.seed)  # for reproduction
    config.save_path = set_save_path(config)

    if config.mode == 'train':
        print("train mode!")
        trainer = Trainer(config)
        trainer.train(config)
    else:
        print("test mode!")
        tester = Tester(config)
        tester.test(config)

if __name__ == '__main__':
    main()
