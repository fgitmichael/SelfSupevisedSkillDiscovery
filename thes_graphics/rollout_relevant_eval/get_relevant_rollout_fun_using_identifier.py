from thes_graphics.rollout_relevant_eval.max_abs_x import extract_max_abs_x


def get_relevant_rollout_fun_using_identifier(identifier: str):
    if identifier == 'max_abs_x':
        return extract_max_abs_x

    else:
        raise NotImplementedError
