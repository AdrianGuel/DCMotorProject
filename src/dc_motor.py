from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DCMotorParams:
    R: float   # Armature resistance (ohm)
    L: float   # Armature inductance (H)
    Ke: float  # Back-emf constant (V*s/rad)
    Kt: float  # Torque constant (N*m/A)
    J: float   # Rotor inertia (kg*m^2)
    b: float   # Viscous friction (N*m*s/rad)
    Tl: float = 0.0  # Constant load torque (N*m)
    Tc: float = 0.0  # Coulomb friction torque (N*m)


def dc_motor_state_matrices(
    params: DCMotorParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Continuous-time state-space matrices from Electronics 2024, 13, 2225 (eqs. 7-9):
        x = [ia, omega]^T
        x_dot = A x + B u
        y = C x + D u, with output y = omega.
    """
    A = np.array(
        [
            [-params.R / params.L, -params.Ke / params.L],
            [params.Kt / params.J, -params.b / params.J],
        ],
        dtype=float,
    )
    B = np.array([[1.0 / params.L], [0.0]], dtype=float)
    C = np.array([[0.0, 1.0]], dtype=float)
    D = np.zeros((1, 1), dtype=float)
    return A, B, C, D


def dc_motor_derivatives(
    t: float, state: np.ndarray, params: DCMotorParams, u: float
) -> np.ndarray:
    """
    State derivative for x = [ia, omega]. Uses the linear state-space model; optional load
    and Coulomb friction are added as disturbances when provided.
    """
    A, B, _, _ = dc_motor_state_matrices(params)
    xdot = A @ state + (B[:, 0] * u)

    # Add non-linear friction or load torque as disturbances if specified.
    if params.Tc != 0.0 or params.Tl != 0.0:
        ia, omega = state
        sign_omega = np.tanh(omega / 1e-3)
        xdot[1] -= (params.Tc * sign_omega + params.Tl) / params.J

    return xdot
