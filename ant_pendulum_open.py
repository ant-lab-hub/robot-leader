import numpy as np


def initialize_ants(Nmax):
    '''Initialize the ants. Each ant is represented by a tuple
    (puller, lifter, informed, phi) where puller, lifter and informed
    are booleans and phi is a float.'''
    puller = np.zeros(Nmax, dtype=int)
    lifter = np.zeros(Nmax, dtype=int)
    informed = np.zeros(Nmax, dtype=int)
    phi = np.zeros(Nmax)
    return puller, lifter, informed, phi


def getForces(f0, Nmax, puller, informed, phi, cargo_pos):
    '''Compute the total force on the cargo due to the ants.
    The pullers pull tangent to the cargo CoM, the informed ants
    pull in the direction of the nest.'''
    total_force_pullers = np.array([0.0, 0.0])
    total_force_informed = np.array([0.0, 0.0])

    nest_dir = np.array([1, 0])
    tangent_dir = np.array([-cargo_pos[1], cargo_pos[0]]) / np.linalg.norm(np.array([-cargo_pos[1], cargo_pos[0]]))

    for i in range(Nmax):
        if puller[i]:
            total_force_pullers += np.dot(f0 * np.array(
                [np.cos(i * 2 * np.pi / Nmax + phi[i]), np.sin(i * 2 * np.pi / Nmax + phi[i])]),
                                          tangent_dir) * tangent_dir
        elif informed[i]:
            total_force_informed += np.dot(
                f0 * np.array([np.cos(i * 2 * np.pi / Nmax + phi[i]), np.sin(i * 2 * np.pi / Nmax + phi[i])]),
                np.array([1, 0])) * np.array([1, 0])
    return total_force_pullers, total_force_informed


def Ratt(Kon, puller, lifter, informed, Nmax):
    '''Compute the attachment rate. The attachment rate
    is proportional to the number of empty sites.'''
    empty_sites = [1 for i in range(Nmax) if puller[i] == lifter[i] == informed[i] == 0]
    return Kon * sum(empty_sites)


def Rdet(Koff, puller, lifter, Nmax):
    '''Compute the detachment rate. The detachment rate
    is proportional to the number of attached puller and lifetr ants.'''
    return Koff * (sum(puller) + sum(lifter))


def Rorient(Kori, puller, informed, Nmax):
    '''Compute the reorientation rate. The reorientation rate
    is proportional to the number of attached puller ants.'''
    return Kori * (sum(puller) + sum(informed))


def Rcon(Kc, puller, lifter, f_ind, Nmax, phi, Ftot):
    '''Compute the contraction rate. The contraction rate
    is proportional to the number of puller and lifter ants.'''
    return Kc * sum([puller[i] * exp_rate(i, f_ind, Nmax, phi, Ftot) + lifter[i] / exp_rate(i, f_ind, Nmax, phi, Ftot)
                     for i in range(Nmax) if puller[i] == 1 or lifter[i] == 1])


def Rfor(Kfor, informed, Nmax):
    '''Compute the forgetting rate. The forgetting rate
    is proportional to the number of informed ants.'''
    return Kfor * sum(informed)


def attach(i, f_ind, puller, lifter, informed, Nmax, phi, Ftot):
    '''Attach an ant to the cargo. The newly attached ant is an informed puller.
    After attachment, she reorients. If the ant is already attached, she forgets.'''
    if puller[i] == 1:
        forget(i, f_ind, Nmax, puller, lifter, informed, phi)
    else:
        informed[i] = 1
        reorient(i, puller, informed, Nmax, phi, Ftot, phi_max=52 * np.pi / 180)


def detach(i, puller, lifter, phi, Nmax):
    '''Detach an ant from the cargo. The detached ant is either a puller or a lifter.'''
    puller[i] = 0
    lifter[i] = 0
    phi[i] = np.nan


def reorient(i, puller, informed, Nmax, phi, Ftot, phi_max=52 * np.pi / 180):
    '''Reorient an ant. The ant is either a puller or an informed ant.
    Puller ants reorient along the direction of the total force on the cargo.
    Informed ants reorient in the direction of the nest.'''
    chi = 1e-6
    force_dir = Ftot / (np.linalg.norm(Ftot) + chi)
    nest_vec = np.array([1, 0])
    site_vec = np.array([np.cos(i * 2 * np.pi / Nmax), np.sin(i * 2 * np.pi / Nmax)])

    if puller[i]:
        force_dir_vec = np.array([np.cos(np.arctan2(force_dir[1], force_dir[0])),
                                  np.sin(np.arctan2(force_dir[1], force_dir[0]))])
        angle = np.arctan2(np.linalg.det([site_vec, force_dir_vec]), np.dot(site_vec, force_dir_vec))
        angle = (angle + 2 * np.pi) % (2 * np.pi)

        if angle > phi_max and angle < (2 * np.pi - phi_max):
            angle = phi_max if angle < np.pi else (2 * np.pi - phi_max)

        phi[i] = angle

    elif informed[i]:
        angle = np.arctan2(np.linalg.det([site_vec, nest_vec]), np.dot(site_vec, nest_vec))
        angle = (angle + 2 * np.pi) % (2 * np.pi)

        if angle > phi_max and angle < (2 * np.pi - phi_max):
            angle = phi_max if angle < np.pi else (2 * np.pi - phi_max)

        phi[i] = angle

    return phi


def pull_or_lift(i, f_ind, Nmax, phi, Ftot):
    '''Determine whether a site is a puller or a lifter.'''
    r3 = np.random.uniform(0, 1)
    prob_pull = 1 / (1 + exp_rate(i, f_ind, Nmax, phi, Ftot))
    if r3 < prob_pull:
        ant_type = 'puller'
    else:
        ant_type = 'lifter'
    return ant_type


def exp_rate(i, f_ind, Nmax, phi, Ftot):
    '''Compute the exponential rate at a site.'''
    pi = np.array([np.cos(phi[i]), np.sin(phi[i])])
    return np.exp(-np.dot(Ftot, pi) / f_ind)


def convert(i, Nmax, puller, lifter, informed, phi, Ftot):
    '''Convert a puller to a lifter or vice versa. If the ant is a puller, she reorients.'''
    puller[i], lifter[i] = lifter[i], puller[i]

    if lifter[i]:
        phi[i] = 0
    else:
        reorient(i, puller, informed, Nmax, phi, Ftot, phi_max=52 * np.pi / 180)


def forget(i, f_ind, Nmax, puller, lifter, informed, phi, Ftot):
    '''An informed ant forgets and then she decides to pull or lift based on a probability.'''
    if informed[i] == 1:
        informed[i] = 0
        ant_type = pull_or_lift(i, f_ind, Nmax, phi, Ftot)
        if ant_type == 'puller':
            puller[i] = 1
            reorient(i, puller, informed, Nmax, phi, Ftot, phi_max=52 * np.pi / 180)
        elif ant_type == 'lifter':
            lifter[i] = 1
            phi[i] = 0


def update_rates(Kon, Koff, Kc, Kori, Kfor, puller, lifter, informed, Nmax, phi, Ftot, f_ind):
    '''Update the event rates for the system.'''
    R_attach = Ratt(Kon, puller, lifter, informed, Nmax)
    R_detach = Rdet(Koff, puller, lifter, Nmax)
    R_convert = Rcon(Kc, puller, lifter, f_ind, Nmax, phi, Ftot)
    R_orient = Rorient(Kori, puller, informed, Nmax)
    R_forget = Rfor(Kfor, informed, Nmax)
    R_total = R_attach + R_detach + R_convert + R_orient + R_forget
    return R_attach, R_detach, R_convert, R_orient, R_forget, R_total


def next_dt_event(R_total):
    '''Compute the time until the next Gillespie event.'''
    r1 = np.random.uniform(0, 1)
    return (- 1 / R_total) * np.log(r1)


def next_event(Kon, Koff, Kc, Kori, Kfor, puller, lifter, informed, Nmax, phi, Ftot, f_ind):
    '''Determine the next event.'''
    r2 = np.random.uniform(0, 1)
    R_attach, R_detach, R_convert, R_orient, R_forget, R_total = update_rates(Kon, Koff, Kc, Kori, Kfor, puller, lifter,
                                                                              informed, Nmax, phi, Ftot, f_ind)

    if r2 < R_attach / R_total:
        '''An ant attaches to a site.'''
        i = np.random.choice([i for i in range(Nmax) if puller[i] == 0 and lifter[i] == 0 and informed[i] == 0])
        attach(i, f_ind, puller, lifter, informed, Nmax, phi, Ftot)

    elif R_attach / R_total <= r2 < (R_attach + R_detach) / R_total:
        '''An ant detaches from a site.'''
        i = np.random.choice([i for i in range(Nmax) if puller[i] == 1 or lifter[i] == 1])
        detach(i, puller, lifter, phi, Nmax)

    elif (R_attach + R_detach) / R_total <= r2 < (R_attach + R_detach + R_convert) / R_total:
        '''An ant converts from a puller to a lifter or vice versa.'''
        indices = [i for i in range(Nmax) if puller[i] == 1 or lifter[i] == 1]
        rates = [
            Kc * (puller[i] * exp_rate(i, f_ind, Nmax, phi, Ftot) + lifter[i] / exp_rate(i, f_ind, Nmax, phi, Ftot))
            for i in
            indices]
        probs = [rate / np.sum(rates) for rate in rates]
        i = np.random.choice(indices, 1, p=np.array(probs))[0]
        convert(i, Nmax, puller, lifter, informed, phi, Ftot)

    elif (R_attach + R_detach + R_convert) / R_total <= r2 < (R_attach + R_detach + R_convert + R_orient) / R_total:
        '''An ant reorients.'''
        i = np.random.choice(np.where(np.array(puller) + np.array(informed))[0])
        reorient(i, puller, informed, Nmax, phi, Ftot)

    elif R_forget != 0:
        '''An ant forgets.'''
        i = np.random.choice(np.where(informed)[0])
        forget(i, f_ind, Nmax, puller, lifter, informed, phi, Ftot)

    dt_event = next_dt_event(R_total)
    return dt_event
