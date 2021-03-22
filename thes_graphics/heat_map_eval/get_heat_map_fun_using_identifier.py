from thes_graphics.heat_map_eval.heat_map_covered_distance_eval import calc_covered_dist_heat_map


def get_heat_map_fun_using_identifier(identifier: str):
    if identifier == 'coverd_dist':
        return calc_covered_dist_heat_map