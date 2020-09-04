import matplotlib.pyplot as plt
import numpy as np

from seqwise_cont_skillspace.utils.get_colors import get_colors


class ModepostPlotter:
    def __init__(self, limits: list=None):
        if limits is None:
            self.limits = [-3, 3]

        self.colors = get_colors()

        self.batch_dim = 0
        self.data_dim = -1
        self.batch_size = 0

    def _check_input(self, skills, mu_post, ids):
        self.batch_size = skills.shape[self.batch_dim]

        assert skills.shape[self.batch_dim] \
               == mu_post.shape[self.batch_dim] \
               == ids.shape[self.batch_dim]
        assert skills.shape[self.data_dim] \
               == mu_post.shape[self.data_dim] \
               == 2, \
            'Plot with other than dimension 2 is not implemented'
        assert ids.shape[self.data_dim] == 1

    def plot_mode_post(
            self,
            skills: np.ndarray,
            mu_post: np.ndarray,
            ids: np.ndarray,
            labels=None,
            set_lim=True,
    ):
        """
        Args:
            skills              : (N, skill_dim)
            mu_post             : (N, skill_dim)
            ids                 : (N, 1) identifier for the skills
                                  (used for legend)
        where skill_dim == 2 is required

        Return:
            figure
        """
        ids = ids.astype(np.int)
        self._check_input(skills, mu_post, ids)

        plt.interactive(False)
        _, axes = plt.subplots()
        if set_lim:
            axes.set_ylim(self.limits)
            axes.set_xlim(self.limits)

        for skill_id in range(np.max(ids) + 1):
            bool_idx = ids == skill_id
            bool_idx = bool_idx.squeeze()
            assert np.all(
                skills[bool_idx]
                == np.stack([skills[bool_idx][0]] * np.sum(bool_idx.astype(np.int)),
                            axis=self.batch_dim))
            plt.scatter(mu_post[bool_idx, 0],
                        mu_post[bool_idx, 1],
                        label="skill {}, {}".format(skill_id, skills[bool_idx][0]),
                        c=self.colors[skill_id])

        axes.legend()
        axes.grid(True)

        return plt.gcf()
