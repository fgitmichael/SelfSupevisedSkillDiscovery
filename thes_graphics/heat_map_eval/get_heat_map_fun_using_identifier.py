from thes_graphics.heat_map_eval.heat_map_covered_distance_eval \
    import calc_covered_dist_heat_map
from thes_graphics.heat_map_eval.heat_map_covered_y_eval import calc_covered_y_heat_map
from thes_graphics.heat_map_eval.heat_map_max_x_val import calc_x_max_heat_map
from thes_graphics.heat_map_eval.heat_map_min_x_val import calc_x_min_heat_map
from thes_graphics.heat_map_eval.heat_map_max_y_val import calc_y_max_heat_map


def get_heat_map_fun_using_identifier(identifier: str):
    if identifier == 'covered_dist':
        return calc_covered_dist_heat_map

    elif identifier == 'covered_y':
        return calc_covered_y_heat_map

    elif identifier == 'max_x':
        return calc_x_max_heat_map

    elif identifier == 'min_x':
        return calc_x_min_heat_map

    elif identifier == 'max_y':
        return calc_y_max_heat_map

    else:
        raise NotImplementedError