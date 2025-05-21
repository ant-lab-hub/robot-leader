"""
The script explores how the ant pendulum responds (Δv) to a grid of force
magnitudes and pulse durations. Results are written to *output.csv*.
"""
import numpy as np               
import pandas as pd             
from scipy.integrate import odeint  
from itertools import product


# -----------------------------------------------------------------------------
# Global simulation settings
# -----------------------------------------------------------------------------
T = [0, 50]        # Time window [s] for every simulation
G = 0.3            # Scaled informed ant force [cm/s]
l = 13.5           # Stiff rod length [cm]
N = 1.75           # Scaled puller ant force [cm/s]
k_c = 0.7          # Conversion rate [1/s]
f_ind = 0.428      # Individuality [cm/s]

params = {
    'G': G,
    'l': l,
    'N': N,
    'k_c': k_c,
    'f_ind': f_ind,
}

# -----------------------------------------------------------------------------
# ODE definition
# -----------------------------------------------------------------------------

def ant_pendulum(y, t, G, l, N, k_c, f_ind, f, dfdt1, dfdt2):
    """Right‑hand side of the ant pendulum ODE.

    Parameters
    ----------
    y : tuple(theta, v)
        theta  – angular position (rad)
        v      – tangential velocity (rad/s)
    t : float
        Current time [s] (passed in automatically by odeint)
    G, l, N, k_c, f_ind : float
        Model parameters (see global block above)
    f : callable
        External force pulse f(t)        ┐
    dfdt1, dfdt2 : callable              │  simple ϵ‑wide top‑hat derivatives
        Smoothed "delta" functions       ┘  for force onset (+) and offset (−)

    Returns
    -------
    dtheta_dt, dv_dt : floats
        Instantaneous derivatives dθ/dt and dv/dt.
    """
    theta, v = y  # unpack state vector

    # Angular kinematics -------------------------------------------------------
    dtheta_dt = v / l

    informed_ants   = -(G * v * np.cos(theta) / l)
    puller_ants      = N * k_c * np.sinh(v / f_ind)
    role_switch   = -2 * k_c * np.cosh(v / f_ind) * (v + G * np.sin(theta) + f(t))
    impulses  = -(dfdt1(t) - dfdt2(t))

    dv_dt = informed_ants + puller_ants + role_switch + impulses
    return dtheta_dt, dv_dt

# -----------------------------------------------------------------------------
# Helper: integrate dynamics for one external force specification
# -----------------------------------------------------------------------------

def simulate_pendulum(T, params, f, dfdt1, dfdt2):
    """Integrate the ODE over *T* for a particular external force.

    Returns a Pandas DataFrame with columns:
        time, q (=θ), qDot (=v)
    """
    ic = [-np.pi / 2, 0]  # initial condition: 90° below pivot, zero velocity
    t = np.linspace(T[0], T[1], 25 * (T[1] - T[0]))  # 25 time‑steps per second

    
    y = odeint(
        ant_pendulum,
        ic,
        t,
        args=(params['G'], params['l'], params['N'], params['k_c'],
              params['f_ind'], f, dfdt1, dfdt2),
    )

    # Pack into DataFrame for convenience
    return pd.DataFrame({'time': t, 'q': y[:, 0], 'qDot': y[:, 1]})

# -----------------------------------------------------------------------------
# Helper: detect 0‑crossings of θ  ==> times pendulum passes the "nest"
# -----------------------------------------------------------------------------

def nestCrossings(t, y):
    """Return time stamps where the sign of θ changes (i.e., pendulum crosses 0)."""
    sign_switch = np.abs(np.diff(np.sign(y))).astype(bool)
    return t[:-1][sign_switch] + np.diff(t)[sign_switch] / (
        np.abs(y[1:][sign_switch] / y[:-1][sign_switch]) + 1
    )

# -----------------------------------------------------------------------------
# Main experiment: grid‑search over force magnitude & pulse duration
# -----------------------------------------------------------------------------

def stepForcingFun(T, params):
    """Pulse the pendulum once and quantify Δv for a range of (Δt, F).

    1. Run a baseline simulation with no external force to find the
       first nest‑crossing time t0.
    2. For each combination of pulse width Δt and magnitude F_ext,
       run the simulation again, apply a rectangular pulse beginning at t0,
       and measure how the mean velocity 3s after the pulse compares to the
       baseline 3s before the pulse.
    """

    # 1) Baseline ----------------------------------------------------------------
    df_baseline = simulate_pendulum(T, params, lambda t: 0, lambda t: 0, lambda t: 0)
    t0 = nestCrossings(df_baseline['time'].values, df_baseline['q'].values)[0]

    # 2) Parameter grid ---------------------------------------------------------
    Dt      = np.linspace(0, 10, 100)   # pulse widths [s]
    F_ext   = np.linspace(0, 1, 100)    # pulse magnitudes [dimensionless]
    grid    = list(product(Dt, F_ext))  # 10000 combinations

    response_data = []

    for Δt, F in grid:
        # Build *closure* functions capturing the pair (Δt,F) --------------
        f      = lambda t, Δt=Δt, F=F: F if t0 < t < t0 + Δt else 0
        # Smoothed on/off edges (width ε) so derivative exists --------------
        ε      = 0.001
        dfdt1  = lambda t, Δt=Δt, F=F, ε=ε: F if t0 - ε < t < t0 + ε else 0
        dfdt2  = lambda t, Δt=Δt, F=F, ε=ε: F if t0 + Δt - ε < t < t0 + Δt + ε else 0

        # 3) Run perturbed simulation -------------------------------------
        df_pert = simulate_pendulum(T, params, f, dfdt1, dfdt2)

        # Window of interest: up to 10s after pulse onset -----------------
        window = df_pert[(df_pert['time'] >= t0) & (df_pert['time'] <= t0 + 10)]

        # Identify first zero‑crossing of velocity sign post‑pulse ---------
        zero_cross = window.loc[(np.sign(window['qDot']).diff().ne(0))]
        idx_first  = np.argmax(zero_cross['qDot'].lt(0).to_numpy())
        zero_cross = zero_cross.reset_index(drop=True)

        # 4) Compute mean velocities before & after ------------------------
        v_before = df_baseline.loc[
            (df_baseline['time'] >= t0 - 3) & (df_baseline['time'] <= t0), 'qDot']
        v_before = v_before.mean()

        if idx_first == 0:
            # Pulse *did not* flip sign before 10s window
            v_after = df_baseline.loc[
                (df_baseline['time'] >= t0 + Δt) & (df_baseline['time'] <= t0 + Δt + 3), 'qDot']
            v_after = v_after.mean()
            Δv = v_after - v_before
            response_data.append([F, Δt, Δv, idx_first])
        else:
            # Pulse induced a sign change; measure after that crossing
            t_cross = zero_cross['time'][1]
            v_after = df_baseline.loc[
                (df_baseline['time'] >= t_cross) & (df_baseline['time'] <= t_cross + 3), 'qDot']
            v_after = v_after.mean()
            Δv = v_after - v_before
            response_data.append([F, t_cross - t0, Δv, idx_first])

    return response_data

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    responses = stepForcingFun(T, params)
    df_out = pd.DataFrame(responses, columns=['force', 'dt', 'dv', 'response'])
    df_out.to_csv('output.csv', index=False)
    print('Sweep finished → results written to output.csv')
