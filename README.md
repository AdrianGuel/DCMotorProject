# DC Motor Control (Paper-based Model)

This project simulates and controls a DC motor using the state-space model and controller tunings published in *Electronics 2024, 13, 2225* (“modeling, measurement, identification and control of direct current motors…”). The default motor parameters in `config.json` match the identified plant in that paper.

## Quick start
- Python 3.10+ recommended. Install deps: `pip install -r requirements.txt` (or activate the existing `.venv`).
- Run a simulation and generate plots: `python main.py`
- View results: open `dc_motor_response.html` in a browser.

## Configuration
Edit `config.json` to change scenarios:
- `motor`: physical parameters (currently set to paper values).
- `simulation`: start conditions, supply voltage, step size `dt` (controller sample), horizon `t_end`, and integrator method (`rk4` or `euler`).
- `control`:
  - `mode`: `"closed"` or `"open"`.
  - `controller`: `"pid"` or `"state_feedback"`.
  - `setpoint`: speed target in rad/s.
  - `pid`:
    - `form`: `"paper"` uses the paper’s discrete-time difference equations (Tables 4 & 5) with presets like `paper_pid_fs50`, `paper_pi_fs50`, etc.
    - `structure`: `"p"`, `"pi"`, `"pd"`, `"pid"` selects which preset coefficients apply.
    - `T`: sampling time (seconds); align with `simulation.dt` when using paper presets.
  - `state_feedback`: `gain_set` picks pole-placement/LQ gains from the paper (`paper_case1`, `paper_case2`, `paper_lq`).
  - `u_min`/`u_max`: actuator saturation in volts.

## Controllers implemented
- **Paper PID (default)**: IIR difference equation `u[k] = a1 u[k-1] + a2 u[k-2] + b0 e[k] + b1 e[k-1] + b2 e[k-2]`, coefficients from the paper for 50 Hz and 100 Hz sampling.
- **Incremental PID (legacy)**: q0/q1/q2 form with `kp/ki/kd` if you set `form: "incremental"`.
- **State feedback with integrator**: `u = -K [ia, omega, xi]`, where `xi` integrates speed error; gain presets mirror the paper’s pole-placement and LQ results.

## Outputs
`dc_motor_response.html` plots:
1. Armature current `ia`
2. Angular velocity `omega` with optional setpoint trace
3. Input voltage `u`

## Citation
If you use this code, please cite the source model and controller design from:
> Electronics 2024, 13, 2225. “Modeling, measurement, identification and control of direct current motors …”

## Notes
- The paper’s controller gains assume the identified plant and the stated sampling times. If you change motor parameters or `dt`, retune or select appropriate presets to maintain stability.
- Load torque and Coulomb friction are optional disturbances; set `Tl`/`Tc` in `motor` if needed.
