import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from tqdm import tqdm

from ce_vae_test.networks.ce_vae import MinVae

from mode_disent.utils.mmd import compute_mmd_tutorial
from code_slac.utils import calc_kl_divergence
from code_slac.utils import update_params

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class CeVaeTrainer(object):

    def __init__(self,
                 vae: MinVae,
                 num_epochs,
                 train_loader,
                 test_loader,
                 writer: SummaryWriter,
                 alpha,
                 lamda):
        self.vae = vae
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.ce_criterion = torch.nn.CrossEntropyLoss()

        self.alpha = alpha
        self.lamda  =lamda

        self.writer = writer

        self.optimizer = torch.optim.Adam(
            vae.parameters(),
            lr=0.0001
        )

    def loss(self, step, data, label):
        self.vae.train()
        forward_return_dict = self.vae(data)
        latent_post = forward_return_dict['latent_post']
        latent_pri = forward_return_dict['latent_pri']
        score = forward_return_dict['recon']

        assert score.shape == torch.Size((data.size(0), self.vae.output_size))

        # KLD
        kld = calc_kl_divergence([latent_post['dist']],
                                 [latent_pri['dist']])

        # MMD
        mmd = compute_mmd_tutorial(latent_post['sample'],
                                   latent_pri['sample'])

        # CE loss
        ce_loss = self.ce_criterion(score, label)

        # Info-VAE loss
        alpha = self.alpha
        lamda = self.lamda
        kld_info = (1 - alpha) * kld
        mmd_info = (alpha + lamda - 1) * mmd
        info_loss = ce_loss + kld_info + mmd_info

        loss_on_latent = kld_info + mmd_info
        loss_on_data  = ce_loss
        loss_latent_minus_data = loss_on_latent - data

        info_loss_prestring = 'info_loss'
        self.log(
            step,
            {
                'hp/alpha': alpha,
                'hp/lamda': lamda,
                'ce_loss': ce_loss,
                '{}/kld'.format(info_loss_prestring): kld,
                '{}/mmd'.format(info_loss_prestring): mmd,
                '{}/mmd_info_weighted'.format(info_loss_prestring): mmd_info,
                '{}/kld_info_weighted'.format(info_loss_prestring): kld_info,
                '{}/info_loss'.format(info_loss_prestring): info_loss,
                '{}/loss_on_latent'.format(info_loss_prestring): loss_on_latent,
                '{}/loss_on_data'.format(info_loss_prestring): loss_on_data
            }
        )

        return info_loss

    def log(self, epoch, to_log: dict):
        for key, el in to_log.items():
            self.writer.add_scalar(
                tag=key,
                scalar_value=el,
                global_step=epoch
            )

    def train(self, step, data, label):
        self.vae.train(True)
        loss = self.loss(step, data, label)
        update_params(self.optimizer, self.vae, loss)

    @torch.no_grad()
    def test(self, epoch, step, axes, fig, data, label):
        self.vae.train(False)
        forward_return_dict = self.vae(data)
        latent_post = forward_return_dict['latent_post']
        score = forward_return_dict['recon']

        self.write_mode_map(epoch, step, axes, fig, latent_post, label)

        self.write_accuracy(step, score, label)

    def write_accuracy(self, epoch, score, label):
        pred_logsoftmax = F.log_softmax(score, dim=-1)
        pred = torch.argmax(pred_logsoftmax, dim=-1)
        accuracy = torch.sum(torch.eq(pred, label)).float()/score.size(0)

        self.writer.add_scalar(
            tag='accuracy',
            scalar_value=accuracy,
            global_step=epoch
        )

        return accuracy

    def write_mode_map(self, epoch, step, axes, fig, latent_post, label):
        mu = latent_post['sample']
        assert mu.shape[-1] == 2

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k',
                  'darkorange', 'gray', 'lightgreen']

        lim = [-3., 3.]
        axes.set_ylim(lim)
        axes.set_xlim(lim)

        for number in range(10):
            bool_idx = label == number
            plt.scatter(mu.detach().cpu().numpy()[bool_idx.cpu().numpy(), 0],
                        mu.detach().cpu().numpy()[bool_idx.cpu().numpy(), 1],
                        label=number,
                        c=colors[number])

        #axes.legend()
        axes.grid(True)
        self.writer.add_figure(
            tag='epoch{}/mode_map'.format(epoch),
            figure=fig,
            global_step=step
        )

    def run(self):
        num_episodes = 10
        fig, axes = plt.subplots()
        step = 0
        for epoch in tqdm(range(self.num_epochs)):
            train_iter = iter(self.train_loader)
            test_iter = iter(self.test_loader)
            for episode in range(num_episodes):

                train_data, train_label = train_iter.__next__()
                self.train(
                    step=step,
                    data=train_data.to(self.vae.device).reshape(-1, 28 * 28),
                    label=train_label.to(self.vae.device),
                )

                test_data, test_label = test_iter.__next__()
                plt.clf()
                self.test(
                    epoch=epoch,
                    step=step,
                    data=test_data.to(self.vae.device).reshape(-1, 28 * 28),
                    label=test_label.to(self.vae.device),
                    fig=fig,
                    axes=axes,
                )

                step += 1
