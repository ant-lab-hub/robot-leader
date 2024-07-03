import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from itertools import product
import matplotlib.cm as cm
import matplotlib.colors as colors
import random

T = [0, 50]
G = 0.3
l = 13.5
N = 1.75
k_c = 0.7
f_ind = 0.428

params = {
    'G': G,
    'l': l,
    'N': N,
    'k_c': k_c,
    'f_ind': f_ind,
}


def ant_pendulum(y, t, G, l, N, k_c, f_ind, f, dfdt1, dfdt2):
    theta, v = y
    dtheta_dt = v / l
    dv_dt = -(G * v * np.cos(theta) / l) + (N * k_c * np.sinh(v / f_ind)) - 2 * k_c * np.cosh(v / f_ind) * (
            v + G * np.sin(theta) + f(t)) - (dfdt1(t) - dfdt2(t))
    return dtheta_dt, dv_dt


def simulate_pendulum(T, params, f, dfdt1, dfdt2):
    ic = [-np.pi / 2, 0]
    t = np.linspace(T[0], T[1], 25 * (T[1] - T[0]))
    y = odeint(ant_pendulum, ic, t,
               args=(params['G'], params['l'], params['N'], params['k_c'], params['f_ind'], f, dfdt1, dfdt2))
    df = pd.DataFrame({'time': t, 'q': y[:, 0], 'qDot': y[:, 1]})
    return df


def nestCrossings(t, y):
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return t[:-1][s] + np.diff(t)[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)

def stepForcingFun(T, params):
    df = simulate_pendulum(T, params, lambda t: 0, lambda t: 0, lambda t: 0)
    t0 = nestCrossings(df['time'].values, df['q'].values)[0]

    Dt = np.linspace(0, 10, 100)
    f_ext = np.linspace(0, 1, 100)
    M = list(product(Dt, f_ext))

    response_data = []

    for m in M:
        f = lambda t, m=m: m[1] if t0 < t < t0 + m[0] else 0
        dfdt1 = lambda t, m=m, e=0.001: m[1] if t0 - e < t < t0 + e else 0
        dfdt2 = lambda t, m=m, e=0.001: m[1] if t0 + m[0] - e < t < t0 + m[0] + e else 0

        df_ext = simulate_pendulum(T, params, f, dfdt1, dfdt2)
        df_crossings = df_ext[(df_ext['time'] >= t0) & (df_ext['time'] <= t0 + 10)]
        df_crossings = df_crossings.loc[(np.sign(df_crossings['qDot']).diff().ne(0))]

        r = np.argmax(df_crossings['qDot'].lt(0).to_numpy(), axis=0)
        df_ = df_crossings.reset_index(drop=True)

        if r == 0:
            v_before = df.loc[(df['time'] >= t0 - 3) & (df['time'] <= t0), 'qDot'].mean()
            v_after = df.loc[(df['time'] >= t0 + m[0]) & (df['time'] <= t0 + m[0] + 3), 'qDot'].mean()
            dv = v_after - v_before
            resData = [m[1], m[0], dv, r]
        else:
            v_before = df.loc[(df['time'] >= t0 - 3) & (df['time'] <= t0), 'qDot'].mean()
            v_after = df.loc[(df['time'] >= df_['time'][1]) & (df['time'] <= df_['time'][1] + 3), 'qDot'].mean()
            dv = v_after - v_before
            resData = [m[1], df_['time'][1] - t0, dv, r]
        response_data.append(resData)

    return response_data


responseData = stepForcingFun(T, params)
df = pd.DataFrame(responseData, columns=['force', 'dt', 'dv', 'response'])
df.to_csv('output.csv', index=False)

