from typing import Dict, List
import numpy as np

from diayn_seq_code_revised.data_collector.seq_collector_revised_discrete_skills import \
    SeqCollectorRevisedDiscreteSkills
from diayn_seq_code_revised.data_collector.rollouter_revised import RollouterRevised
from diayn_seq_code_revised.data_collector.pixel_rollout import pixel_rollout


class SeqCollectorRevisedDiscreteSkillsPixel(SeqCollectorRevisedDiscreteSkills):

    def create_rollouter(
            self,
            env,
            policy,
    ):
        return RollouterRevised(
            env=env,
            policy=policy,
            rollout_fun=pixel_rollout,
        )

    def _collect_new_paths(self,
                           num_seqs: int,
                           seq_len: int) -> List[Dict]:
        paths = []
        num_steps_collected = 0

        for _ in range(num_seqs):
            path = self._rollouter.do_rollout(
                seq_len=seq_len
            )

            pixel_obs = path.next_obs[1]
            path.next_obs = path.next_obs[0]

            self._check_paths(
                path=path,
                seq_len=seq_len,
            )
            assert isinstance(pixel_obs, np.ndarray)
            assert len(pixel_obs.shape) == 4
            assert pixel_obs.shape[0] == seq_len

            num_steps_collected += seq_len

            path = dict(
                path_mapping=path,
                pixel_obs=pixel_obs,
            )
            paths.append(path)

        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected

        return paths

    def _check_path(self, path, seq_len):
        path = path['path_mapping']
        super()._check_path(path, seq_len)

    def prepare_paths_before_save(self, paths: List[dict], seq_len) -> List[Dict]:
        path_mappings = [path['path_mapping'] for path in paths]
        pixel_obs = [path['pixel_obs'] for path in paths]
        prepared_path_mappings = super().prepare_paths_before_save(
            paths=path_mappings,
            seq_len=seq_len,
        )

        prepared_paths = []
        for path_mapping, pixel_o in zip(prepared_path_mappings, pixel_obs):
            prepared_path = dict(
                **path_mapping,
                pixel_obs=pixel_o,
            )
            prepared_paths.append(prepared_path)

        return prepared_paths

    def get_epoch_paths(self) -> List[dict]:
        """
        Return:
            list of dicts with keys of TransitionModeMappingDiscreteSkills and pixel_obs
        """
        assert len(self._epoch_paths) > 0

        epoch_paths = list(self._epoch_paths)
        epoch_paths_ret = []
        for idx, path in enumerate(epoch_paths):
            assert len(path['obs'].shape) == 2

            epoch_path_ret = {}
            for k, v in path.items():
                # tranpose the usual elements and don't transpose pixel obs
                if isinstance(v, np.ndarray) and len(v.shape) == 2:
                    epoch_path_ret[k] = path[k].transpose(1, 0)
                else:
                    epoch_path_ret[k] = path[k]

            epoch_paths_ret.append(epoch_path_ret)
            self.reset()

        return epoch_paths_ret
