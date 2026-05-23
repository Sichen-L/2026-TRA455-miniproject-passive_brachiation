# Passive Brachiation Framework

This folder contains a simplified two-link inverted-pendulum framework for the
project "An Inverted Compass Gait Model for Passive Brachiation".

The model uses four state variables:

```text
q1      support-link angle from vertical down
q2      middle-joint relative angle
q1_dot  support-link angular velocity
q2_dot  middle-joint angular velocity
```

The free endpoint can contact an inclined slope surface.  A caller-provided
policy decides whether the model switches support hands:

```python
def switch_policy(t, state, support_point, impact_point, slope):
    return SwitchDecision.SWITCH
```

Possible decisions:

- **SWITCH** - release the old support; the collision point becomes the new support
- **NO_SWITCH** - release the free end and keep swinging from the old support
- **DWELL** - double-support rest with zero velocity

Collision handling is selectable.  The default is the full-grab 1-D model,
matching the `passive-brachiator-main` KKT impact map:

```python
samples = simulate(
    model,
    slope,
    initial_state,
    initial_support_point,
    duration=2.0,
    dt=0.005,
    collision_mode=CollisionMode.FULL_GRAB_1D,
)
```

You can pass a custom collision function later for a 2-D shooting model:

```python
def my_collision_model(context):
    ...
    return SwitchResult(state=new_state, phase=SwitchDecision.SWITCH)

samples = simulate(..., collision_model=my_collision_model)
```

Shooting utilities live in `shooting.py`.  The Newton solver is generic and
only needs a stride map `P(x)`:

```python
from passive_brachiation import (
    grid_search_initial_guesses,
    make_passive_brachiation_feasibility_check,
    make_passive_brachiation_stride_map,
    newton_shoot,
)

P = make_passive_brachiation_stride_map(model, slope, dt=0.005)
check = make_passive_brachiation_feasibility_check(model.p, dim=1)
guesses = grid_search_initial_guesses(P, bounds=[(0.05, 0.6)], grid_sizes=20, feasibility_check=check)
result = newton_shoot(P, guesses[0].x, feasibility_check=check)
```

The dynamics also expose default-zero disturbance interfaces:

```python
params = BrachiationParameters.uniform_links(damping1=0.0, damping2=0.0)

def endpoint_force_policy(t, state):
    return np.zeros(2)  # [Fy, Fz] at the free endpoint

def generalized_force_policy(t, state):
    return np.zeros(2)  # [Q1, Q2] in generalized coordinates
```

Each `SimulationSample` records elbow torque, endpoint force, generalized
force, kinetic energy, potential energy, and total mechanical energy.

Run the terminal demo from the project root:

```powershell
python -m passive_brachiation.demo
```

Main files:

- `model.py`: parameters, state, mass matrix, gravity, acceleration, RK4 step.
- `kinematics.py`: coordinate convention, slope definition, forward kinematics.
- `switching.py`: slope-impact detection, collision models, support switching.
- `simulation.py`: fixed-step simulation loop with pluggable collision handling.
- `shooting.py`: generic Newton/grid-search shooting and passive stride-map factory.
- `demo.py`: minimal runnable example.
