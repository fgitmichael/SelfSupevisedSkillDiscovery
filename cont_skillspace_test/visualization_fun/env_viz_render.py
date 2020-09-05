import time

from cont_skillspace_test.visualization_fun.env_viz_base import \
    EnvVisualizationGuidedBase, EnvVisualizationHduvaeBase


class EnvVisualizationRenderGuided(EnvVisualizationGuidedBase):

    def __init__(self,
                 *args,
                 render_dt=0.008,
                 **kwargs
                 ):
        super().__init__(
            *args,
            **kwargs
        )
        self.render_dt = render_dt

    def reset(self):
        super().reset()
        self.env.render()

    def visualize(self):
        for _ in range(self.seq_len):
            a, policy_info = self.policy.get_action(self.obs)
            self.env.step(a)
            time.sleep(self.render_dt)
            self.env.render()
