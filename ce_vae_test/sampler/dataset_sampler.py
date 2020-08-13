import torch
import random

class SamplerDatasetWithReplacement(object):

    def __init__(self,
                 dataset,
                 batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        idx = torch.randint(
            low=0,
            high=len(self),
            size=(batch_size,)
        )

        batch_data = []
        batch_label = []
        idx_data = 0
        idx_label = 1
        for el in idx:
            batch_data.append(
                self.dataset[el][idx_data]
            )
            batch_label.append(
                torch.tensor(
                    [self.dataset[el][idx_label]]
                )
            )

        batch_dim = 0
        batch_data = torch.stack(batch_data, dim=batch_dim)
        batch_label = torch.cat(batch_label, dim=batch_dim)

        return (batch_data, batch_label)
