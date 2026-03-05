"""Reusable numerical integration routines."""

from typing import Callable, Dict
import numpy as np


def euler_step(f, t, y, h, *args):
    """Explicit Euler step."""
    return y + h * f(t, y, *args)


def rk4_step(f, t, y, h, *args):
    """Classic fourth-order Runge–Kutta step."""
    k1 = f(t, y, *args)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1, *args)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


METHODS: Dict[str, Callable] = {
    "euler": euler_step,
    "rk4": rk4_step,
}
