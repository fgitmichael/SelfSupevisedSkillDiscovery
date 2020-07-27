from collections import OrderedDict

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict

from rlkit.torch.sac.diayn.diayn import DIAYNTrainer


class DIAYNTrainerExtension(DIAYNTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_samples_trained = 0

    def train_from_torch(self, batch):
        obs = batch['observations']
        batch_size = obs.shape[0]
        self.num_samples_trained += batch_size

        super().train_from_torch(batch)

        self.eval_statistics.update(create_stats_ordered_dict(
            'Number of Samples used for Training',
            self.num_samples_trained
        ))





