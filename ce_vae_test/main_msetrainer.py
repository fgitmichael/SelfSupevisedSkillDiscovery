from __future__ import print_function
import argparse
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

from ce_vae_test.networks.min_vae import MinVae
from ce_vae_test.trainer.mse_trainer_on_latent_no_decoder import MseVaeTrainer
from ce_vae_test.sampler.dataset_sampler import SamplerDatasetWithReplacement

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

writer = SummaryWriter(comment='orig')

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_sampler = SamplerDatasetWithReplacement(
    dataset=datasets.MNIST('../data',
                           train=True,
                           download=True,
                           transform=transforms.ToTensor()),
    batch_size=args.batch_size
)
test_sampler = SamplerDatasetWithReplacement(
    dataset=datasets.MNIST('../data',
                           train=False,
                           transform=transforms.ToTensor()),
    batch_size=args.batch_size * 10
)

vae = MinVae(
    input_size=28 * 28,
    output_size=28 * 28,
    latent_dim=2,
    hidden_sizes_dec=[5],
    device=device
).to(device)

trainer = MseVaeTrainer(
    vae=vae,
    num_epochs=300,
    train_loader=train_sampler,
    test_loader=test_sampler,
    writer=writer,
    device=device,
    alpha=0.99999,
    lamda=0.1,
    gamma=0.0,
)

trainer.run()
