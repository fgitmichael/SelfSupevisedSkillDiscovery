import numpy as np
import matplotlib.pyplot as plt

from diayn_original_cont.utils.mode_post_plotter import ModepostPlotter


class ModepostPlotterHighdimusingvae(ModepostPlotter):

    def _check_input(
            self,
            mu_post,
            ids,
            skills = None,

    ):
        if skills is not None:
            raise ValueError("In the highdim case skills are changing in every iteration,"
                             "while only the latent samples, that are evaluated stay "
                             "the same. Skills are now of higher dim than two. Putting "
                             "them in the legend does not make sence anymore")

        assert mu_post.shape[self.batch_dim] == ids.shape[self.batch_dim]
        assert mu_post.shape[self.data_dim] == 2, \
            'Plot with other than dimension 2 is not implemented'
        assert ids.shape[self.data_dim] == 1

    def plot_mode_post(
            self,
            mu_post: np.ndarray,
            ids: np.ndarray,
            skills: np.ndarray = None,
            labels=None,
            set_lim=True,
    ):
        """
        Args:
            mu_post             : (N, skill_dim)
            ids                 : (N, 1) identifier for the skills
                                  (used for legend)
        where skill_dim == 2 is required

        Return:
            figure
        """
        if skills is not None:
            raise ValueError("In the highdim case skills are changing in every iteration,"
                             "while only the latent samples, that are evaluated stay "
                             "the same. Skills are now of higher dim than two. Putting "
                             "them in the legend does not make sence anymore")

        ids = ids.astype(np.int)
        self._check_input(
            mu_post=mu_post,
            ids=ids,
        )

        plt.interactive(False)
        _, axes = plt.subplots()
        if set_lim:
            axes.set_ylim(self.limits)
            axes.set_xlim(self.limits)

        for skill_id in range(np.max(ids) + 1):
            bool_idx = ids == skill_id
            bool_idx = bool_idx.squeeze()
            plt.scatter(mu_post[bool_idx, 0],
                        mu_post[bool_idx, 1],
                        label="skill {}".format(skill_id),
                        c=self.colors[skill_id])

        axes.legend()
        axes.grid(True)

        return plt.gcf()

