def fix_shift_amount_list(shift_amount_list):
    # compatilibty for sahi v0.8.15
    if isinstance(shift_amount_list[0], (int, float)):
        shift_amount_list = [shift_amount_list]
    return shift_amount_list


def fix_full_shape_list(full_shape_list):
    # compatilibty for sahi v0.8.15
    if full_shape_list is not None and isinstance(full_shape_list[0], (int, float)):
        full_shape_list = [full_shape_list]
    return full_shape_list
