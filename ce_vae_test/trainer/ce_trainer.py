import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from tqdm import tqdm

from ce_vae_test.networks.min_vae import MinVae
from ce_vae_test.sampler.dataset_sampler import SamplerDatasetWithReplacement

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
                 train_loader: SamplerDatasetWithReplacement,
                 test_loader: SamplerDatasetWithReplacement,
                 device,
                 writer: SummaryWriter,
                 alpha,
                 lamda):
        self.num_epochs = num_epochs

        self.ce_criterion = torch.nn.CrossEntropyLoss()

        self.vae = vae
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.alpha = alpha
        self.lamda = lamda
        self.writer = writer
        self.device = device

        self.optimizer = torch.optim.Adam(
            vae.parameters(),
            lr=0.0001
        )

    def loss_latent(self, step, forward_return_dict) -> torch.Tensor:
        latent_post = forward_return_dict['latent_post']
        latent_pri = forward_return_dict['latent_pri']

        # KLD
        kld = calc_kl_divergence([latent_post['dist']],
                                 [latent_pri['dist']])

        # MMD
        mmd = compute_mmd_tutorial(latent_post['sample'],
                                   latent_pri['sample'])

        alpha = self.alpha
        lamda = self.lamda
        kld_info = (1 - alpha) * kld
        mmd_info = (alpha + lamda - 1) * mmd

        loss_on_latent = kld_info + mmd_info

        pre_string = 'info_loss_latent'
        self.log(
            step,
            {
                'hp/alpha': alpha,
                'hp/lamda': lamda,
                '{}/kld'.format(pre_string): kld,
                '{}/mmd'.format(pre_string): mmd,
                '{}/mmd_info_weighted'.format(pre_string): mmd_info,
                '{}/kld_info_weighted'.format(pre_string): kld_info,
                '{}/loss_on_latent'.format(pre_string): loss_on_latent,
            }
        )


        return loss_on_latent

    def loss_data(self, step, forward_return_dict, data, label) -> torch.Tensor:
        score = forward_return_dict['recon']['sample']

        # CE loss
        ce_loss = self.ce_criterion(score, label)
        loss_on_data  = ce_loss

        prestring = "info_loss_data"
        self.log(
            step,
            {
                'ce_loss': ce_loss,
                '{}/loss_on_data'.format(prestring): loss_on_data
            }
        )

        return loss_on_data

    def loss(self, step, data, label):
        forward_return_dict = self.vae(data)
        score = forward_return_dict['recon']['sample']
        assert score.shape == torch.Size((data.size(0), self.vae.output_size))

        latent_loss = self.loss_latent(
            step=step,
            forward_return_dict=forward_return_dict,
        )

        data_loss = self.loss_data(
            step=step,
            forward_return_dict=forward_return_dict,
            data=data,
            label=label,
        )

        info_loss = data_loss + latent_loss
        loss_latent_minus_data = latent_loss - data_loss

        info_loss_prestring = 'info_loss'
        self.log(
            step,
            {
                '{}/info_loss'.format(info_loss_prestring): info_loss,
                '{}/latent_loss_minus_data'.format(info_loss_prestring):
                    loss_latent_minus_data,
            }
        )

        return info_loss

    def log(self, step, to_log: dict):
        for key, el in to_log.items():
            self.writer.add_scalar(
                tag=key,
                scalar_value=el,
                global_step=step
            )

    def train(self, step, data, label):
        self.vae.train(True)
        loss = self.loss(step, data, label)
        update_params(self.optimizer, self.vae, loss)

    @torch.no_grad()
    def test(self, epoch, step, data, label):
        self.vae.train(False)
        forward_return_dict = self.vae(data)
        latent_post = forward_return_dict['latent_post']
        score = forward_return_dict['recon']['sample']

        self.write_mode_map(epoch, step, latent_post, label)

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

    def create_mode_map(self, latent_post, label, legend=None):
        if legend is not None:
            assert len(legend) == 10
        else:
            legend = 10 * ['']


        mu = latent_post['sample']
        assert mu.shape[-1] == 2

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k',
                  'darkorange', 'gray', 'lightgreen']

        fig, axes = plt.subplots()

        lim = [-3., 3.]
        axes.set_ylim(lim)
        axes.set_xlim(lim)

        for number in range(10):
            bool_idx = label == number
            plt.scatter(mu.detach().cpu().numpy()[bool_idx.cpu().numpy(), 0],
                        mu.detach().cpu().numpy()[bool_idx.cpu().numpy(), 1],
                        label="{}{}".format(number, legend[number]),
                        c=colors[number])

        axes.legend()
        axes.grid(True)

        return plt.gcf()

    def write_mode_map(self, epoch, step, latent_post, label, legend=None):
        fig = self.create_mode_map(
            latent_post=latent_post,
            label=label,
            legend=legend,
        )

        self.writer.add_figure(
            tag='mode_map'.format(epoch),
            figure=plt.gcf(),
            global_step=epoch
        )

        plt.close()

    def run(self):
        num_episodes = 20
        step = 0
        for epoch in tqdm(range(self.num_epochs)):
            for episode in range(num_episodes):

                train_data, train_label = self.train_loader.sample()
                self.train(
                    step=step,
                    data=train_data.to(self.vae.device).reshape(-1, 28 * 28),
                    label=train_label.to(self.vae.device),
                )
                step += 1

            test_data, test_label = self.test_loader.sample()

            self.test(
                epoch=epoch,
                step=step,
                data=test_data.to(self.vae.device).reshape(-1, 28 * 28),
                label=test_label.to(self.vae.device),
            )
