import json
from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dc_motor import DCMotorParams, dc_motor_derivatives
from src.numerical_method import METHODS


def load_config(path: str = "config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cfg = load_config()
    params = DCMotorParams(**cfg["motor"])
    sim = cfg["simulation"]
    ctrl = cfg.get("control", {})

    open_loop_v = float(sim["V"])
    dt = float(sim["dt"])
    t_end = float(sim["t_end"])

    controller = ctrl.get("controller", "pid").lower()
    closed_loop = ctrl.get("mode", "open").lower() == "closed"

    use_state_feedback = closed_loop and controller == "state_feedback"
    state_dim = 3 if use_state_feedback else 2
    state = np.zeros(state_dim, dtype=float)
    state[:2] = np.array([sim["i0"], sim["omega0"]], dtype=float)  # [ia, omega]

    method_name = sim.get("method", "rk4").lower()
    try:
        step = METHODS[method_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown integration method '{method_name}'. Available: {', '.join(METHODS)}"
        ) from exc

    # Controller settings (speed control)
    setpoint = float(ctrl.get("setpoint", 0.0))  # rad/s
    pid_cfg: Dict = ctrl.get("pid", {})
    u_min = ctrl.get("u_min", -24.0)
    u_max = ctrl.get("u_max", 24.0)
    u_prev = float(ctrl.get("u0", 0.0))
    u_prev2 = 0.0
    e_prev = 0.0
    e_prev2 = 0.0

    # PID implementation options
    pid_form = pid_cfg.get("form", "paper").lower()  # "paper" (difference eq) or "incremental"
    pid_structure = pid_cfg.get("structure", "pid").lower()  # "p", "pi", "pd", "pid"

    # Paper discrete-time coefficients (phase margin 60 deg), Tables 4 and 5
    # General form used: u[k] = a1*u[k-1] + a2*u[k-2] + b0*e[k] + b1*e[k-1] + b2*e[k-2]
    paper_coeffs: Dict[str, Dict[str, float]] = {
        # fs = 50 Hz (Ts = 0.02 s)
        "paper_p_fs50":   {"a1": 0.0, "a2": 0.0, "b0": 0.1241, "b1": 0.0, "b2": 0.0},
        "paper_pi_fs50":  {"a1": 1.0, "a2": 0.0, "b0": 0.0642, "b1": -0.0566, "b2": 0.0},
        "paper_pd_fs50":  {"a1": 0.0, "a2": 0.0, "b0": 0.4568, "b1": -0.2705, "b2": 0.0},
        "paper_pid_fs50": {"a1": 1.0, "a2": 0.0, "b0": 0.3577, "b1": -0.5272, "b2": 0.1868},
        # fs = 100 Hz (Ts = 0.01 s)
        "paper_p_fs100":   {"a1": 0.0, "a2": 0.0, "b0": 0.1430, "b1": 0.0, "b2": 0.0},
        "paper_pi_fs100":  {"a1": 1.0, "a2": 0.0, "b0": 0.0720, "b1": -0.0676, "b2": 0.0},
        "paper_pd_fs100":  {"a1": 0.0, "a2": 0.0, "b0": 1.3851, "b1": -1.0659, "b2": 0.0},
        "paper_pid_fs100": {"a1": 1.0, "a2": 0.0, "b0": 1.1911, "b1": -2.0351, "b2": 0.8607},
    }

    if pid_form == "paper":
        default_gain_set = "paper_pid_fs50"
        gain_set = pid_cfg.get("gain_set", default_gain_set).lower()
        coeffs = paper_coeffs.get(gain_set)
        if coeffs is None:
            raise ValueError(f"Unknown paper PID gain_set '{gain_set}'")
        a1 = coeffs["a1"]
        a2 = coeffs["a2"]
        b0 = coeffs["b0"]
        b1 = coeffs["b1"]
        b2 = coeffs["b2"]
    else:
        # Incremental form (legacy)
        pid_presets: Dict[str, Dict[str, float]] = {
            # Continuous-time designs at phase margin 60 deg from the paper (Appendix A)
            "paper_p_phi60": {"kp": 0.1746, "ki": 0.0, "kd": 0.0},
            "paper_pi_phi60": {"kp": 0.0842, "ki": 0.0842 / 0.1588, "kd": 0.0},
            "paper_pd_phi60": {"kp": 0.2605, "ki": 0.0, "kd": 0.2605 * 0.0382},
        }
        pid_gain_set = pid_cfg.get("gain_set")
        if pid_gain_set:
            gains = pid_presets.get(pid_gain_set.lower())
            if gains is None:
                raise ValueError(f"Unknown PID gain_set '{pid_gain_set}'")
            pid_cfg = {**pid_cfg, **gains}  # merge resolved gains

        if {"kp", "ki", "kd"} <= pid_cfg.keys():
            kp = float(pid_cfg["kp"])
            ki = float(pid_cfg["ki"])
            kd = float(pid_cfg["kd"])
            T = float(pid_cfg.get("T", dt))
            q0 = kp + kd / T + ki * T
            q1 = -kp - 2 * kd / T
            q2 = kd / T
        else:
            q0 = float(pid_cfg.get("q0", 0.0))
            q1 = float(pid_cfg.get("q1", 0.0))
            q2 = float(pid_cfg.get("q2", 0.0))

    # State-feedback gains (augmented with integral of speed error)
    sf_cfg: Dict = ctrl.get("state_feedback", {})
    sf_presets: Dict[str, List[float]] = {
        # Derived from paper: eq. (103) - poles {0.8937, 0.5135, 0.5}
        "paper_case1": [4.0835, 0.1610, 0.7598],
        # Derived from paper: eq. (104) - accelerated p1, integrator pole at 0.5
        "paper_case2": [5.9413, 0.3076, 2.8886],
        # LQ design from eq. (105)
        "paper_lq": [2.6691, 0.1270, 0.8351],
    }
    sf_gain_set = sf_cfg.get("gain_set", "paper_case2").lower()
    sf_K = np.array(
        sf_cfg.get("K", sf_presets.get(sf_gain_set, sf_presets["paper_case2"])),
        dtype=float,
    )

    times = np.arange(0.0, t_end + dt, dt)
    traj = np.zeros((len(times), state_dim))
    u_hist = np.zeros(len(times))

    for idx, t in enumerate(times):
        traj[idx] = state
        ia, omega = state[:2]

        if closed_loop:
            if controller == "pid":
                e_k = setpoint - omega
                if pid_form == "paper":
                    u_k = (
                        a1 * u_prev
                        + a2 * u_prev2
                        + b0 * e_k
                        + b1 * e_prev
                        + b2 * e_prev2
                    )
                    u_prev2 = u_prev
                else:
                    u_k = u_prev + q0 * e_k + q1 * e_prev + q2 * e_prev2
                u_k = float(np.clip(u_k, u_min, u_max))
                e_prev2 = e_prev
                e_prev = e_k
                u_prev = u_k
                V_in = u_k
            elif controller == "state_feedback":
                # x = [ia, omega, xi], xi integrates speed error for tracking
                u_k = -float(np.dot(sf_K, state))
                V_in = float(np.clip(u_k, u_min, u_max))
            else:
                raise ValueError(f"Unknown controller '{controller}'")
        else:
            V_in = open_loop_v

        u_hist[idx] = V_in

        if use_state_feedback:
            def augmented_derivatives(t_local, x_local, p, u_local, sp):
                d_ia_omega = dc_motor_derivatives(t_local, x_local[:2], p, u_local)
                d_xi = sp - x_local[1]
                return np.array([d_ia_omega[0], d_ia_omega[1], d_xi], dtype=float)

            state = step(augmented_derivatives, t, state, dt, params, V_in, setpoint)
        else:
            state = step(dc_motor_derivatives, t, state, dt, params, V_in)

    ia = traj[:, 0]
    omega = traj[:, 1]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Armature current ia (A)",
            "Angular velocity omega (rad/s)",
            "Input voltage u (V)",
        ),
    )

    fig.add_trace(
        go.Scatter(x=times, y=ia, mode="lines", name="ia", line=dict(color="steelblue")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=omega, mode="lines", name="omega", line=dict(color="tomato")),
        row=2,
        col=1,
    )
    if closed_loop:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=np.full_like(times, setpoint),
                mode="lines",
                name="setpoint",
                line=dict(color="gray", dash="dash"),
            ),
            row=2,
            col=1,
        )
    fig.add_trace(
        go.Scatter(x=times, y=u_hist, mode="lines", name="u", line=dict(color="seagreen")),
        row=3,
        col=1,
    )

    fig.update_yaxes(title_text="Armature current ia (A)", row=1, col=1)
    fig.update_yaxes(title_text="Angular velocity omega (rad/s)", row=2, col=1)
    fig.update_yaxes(title_text="Input voltage u (V)", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_layout(height=850, width=900, showlegend=True)

    fig.write_html("dc_motor_response.html", include_plotlyjs="cdn")


if __name__ == "__main__":
    main()
