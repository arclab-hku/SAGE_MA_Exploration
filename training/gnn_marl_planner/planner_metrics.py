import math


def update_ema(prev_ema: float, new_value: float, alpha: float) -> float:
    if alpha <= 0.0:
        return prev_ema
    if alpha >= 1.0:
        return new_value
    return (1.0 - alpha) * prev_ema + alpha * new_value


def l2_distance_xy(a_xy, b_xy) -> float:
    dx = float(a_xy[0] - b_xy[0])
    dy = float(a_xy[1] - b_xy[1])
    return float(math.hypot(dx, dy))
