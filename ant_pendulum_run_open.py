import itertools
import numpy as np
from ant_pendulum_open import *


def save_run(data_cargo, ant_state, ant_phi, filename):
    np.savez(filename, data_cargo=data_cargo, ant_state=ant_state, ant_phi=ant_phi)


def run(puller, lifter, informed, phi, Nmax, gamma, total_time, f0=2.8, Kon=0.015, Koff=0.015, Kfor=0.09, Kori=0.7,
        Kc=1, f_ind=28, step_size=0.01):
    '''Run the simulation for a given set of parameters. The simulation is run for a total time of total_time. The step size is 0.01.'''
    total_steps = int(total_time / step_size)
    theta = [0]
    data_cargo = []
    ant_state = np.zeros((Nmax, total_steps), dtype=str)
    ant_phi = np.zeros((Nmax, total_steps))
    rod_length = 1
    cargo_pos = np.array([rod_length, 0], dtype=float)
    Ftot = np.array([0, 0], dtype=float)

    t_total = 0

    for step in range(total_steps):
        ctime = 0
        while ctime < step_size:
            dt_event = next_event(Kon, Koff, Kc, Kori, Kfor, puller, lifter, informed, Nmax, phi, Ftot, f_ind)
            ctime += dt_event
            t_total += dt_event

            Fpullers, Finformed = getForces(f0, Nmax, puller, informed, phi, cargo_pos)
            Ftot = np.array(Fpullers) + np.array(Finformed)

        cargo_pos_new = cargo_pos + (Ftot * step_size) / gamma

        # setting strict boundary condition
        if np.linalg.norm(cargo_pos_new) != rod_length:
            cargo_pos_new = cargo_pos_new / np.linalg.norm(cargo_pos_new) * rod_length

        cargo_pos = cargo_pos_new

        theta[0] = np.arctan2(cargo_pos[1], cargo_pos[0])

        tangent_dir = np.array([-cargo_pos[1], cargo_pos[0]]) / np.linalg.norm(np.array([-cargo_pos[1], cargo_pos[0]]))
        omega = np.dot(Ftot / gamma, tangent_dir) / rod_length

        data_cargo.append(
            {'time': step, 'theta': theta[0], 'omega': omega, 'num_pullers': np.sum(puller),
             'num_lifters': np.sum(lifter),
             'num_informed': np.sum(informed), 'F_pullers': Fpullers, 'F_informed': Finformed})
        ant_state[:, step] = ['p' if p else 'l' if l else 'i' if i else '0'
                              for p, l, i in zip(puller, lifter, informed)]
        ant_phi[:, step] = phi

        t_total += step_size

        if step % (1 / step_size) == 0:
            print(f'Current time: {step * step_size:.2f} s')

    return data_cargo, ant_state, ant_phi

def runSim(total_time):
    '''Run the simulation for a range of Nmax and gamma values.'''
    Nmax = np.linspace(10, 100, 10)
    gamma = 1.77 * Nmax + 2.44

    for n, g in zip(Nmax, gamma):
        n = int(n)
        puller, lifter, informed, phi = initialize_ants(n)
        data_cargo, ant_state, ant_phi = run(puller, lifter, informed, phi, n, g, total_time)
        save_run(data_cargo, ant_state, ant_phi, f'run_{n}_{g}.npz')
        print(f'Run for Nmax={n} and gamma={g} completed.')
    return

runSim(total_time=1000)

# # run the simulation for a fixed Nmax and gamma pair
# puller, lifter, informed, phi = initialize_ants(Nmax=100)
# data_cargo, ant_state, ant_phi = run(puller, lifter, informed, phi, Nmax=100, gamma=180, total_time=1000)
# save_run(data_cargo, ant_state, ant_phi, 'run.npz')

