import numpy as np
import cfd
import cfdsim.lin_coupled
from pyqubo import Array, Constraint
from dwave.samplers import SimulatedAnnealingSampler

def apply_eq_transport_quad(sim: cfd.Simulation, im, Q, A, b, x, y):
    eq_str = ""

    arr_cur_psi_x_y = sim.arr_cur_psi_at(x, y)
    arr_cur_w_x_y = sim.arr_cur_w_at(x, y)

    arr_cur_psi_xp1_y = sim.arr_cur_psi_at(x+1, y)
    if arr_cur_psi_xp1_y is None:
        arr_cur_psi_xp1_y = 0
    arr_cur_w_xp1_y = sim.arr_cur_w_at(x+1, y)
    if arr_cur_w_xp1_y is None:
        arr_cur_w_xp1_y = 0

    arr_cur_psi_x_yp1 = sim.arr_cur_psi_at(x, y+1)
    if arr_cur_psi_x_yp1 is None:
        arr_cur_psi_x_yp1 = 0
    arr_cur_w_x_yp1 = sim.arr_cur_w_at(x, y+1)
    if arr_cur_w_x_yp1 is None:
        arr_cur_w_x_yp1 = 0

    arr_cur_w_xm1_y = sim.arr_cur_w_at(x-1, y)
    if arr_cur_w_xm1_y is None:
        arr_cur_w_xm1_y = 0
    arr_cur_w_x_ym1 = sim.arr_cur_w_at(x, y-1)
    if arr_cur_w_x_ym1 is None:
        arr_cur_w_x_ym1 = 0

    # Crank-Nicolson
    F0 = 0
    F0 += sim.alpha_x*(arr_cur_w_xp1_y - 2*arr_cur_w_x_y + arr_cur_w_xm1_y)
    F0 += sim.alpha_y*(arr_cur_w_x_yp1 - 2*arr_cur_w_x_y + arr_cur_w_x_ym1)
    F0 -= sim.beta*((arr_cur_w_xp1_y - arr_cur_w_x_y)*(arr_cur_psi_x_yp1 - arr_cur_psi_x_y) - (arr_cur_w_x_yp1 - arr_cur_w_x_y)*(arr_cur_psi_xp1_y - arr_cur_psi_x_y))

    b[im] = arr_cur_w_x_y + 0.5*F0

    f_w_x_y = 1 + sim.alpha_x + sim.alpha_y
    if sim.has_bc_at(x, y):
        b[im] -= f_w_x_y*sim.w_bc_val(x, y)
    else:
        A[im, sim.w_nocc_idx(x, y)] = f_w_x_y
        eq_str += f"{f_w_x_y}*w_{x}{y} "

    if (x+1) <= sim.Nx:
        f_w_xp1_y = -0.5*sim.alpha_x
        if sim.has_bc_at(x+1, y):
            b[im] -= f_w_xp1_y*sim.w_bc_val(x+1, y)
        else:
            A[im, sim.w_nocc_idx(x+1, y)] = f_w_xp1_y
            eq_str += f"{f_w_xp1_y}*w_{x+1}{y} "

    if (y+1) <= sim.Ny:
        f_w_x_yp1 = -0.5*sim.alpha_y
        if sim.has_bc_at(x, y+1):
            b[im] -= f_w_x_yp1*sim.w_bc_val(x, y+1)
        else:
            A[im, sim.w_nocc_idx(x, y+1)] = f_w_x_yp1
            eq_str += f"{f_w_x_yp1}*w_{x}{y+1} "

    if (x-1) >= 1:
        f_w_xm1_y = -0.5*sim.alpha_x
        if sim.has_bc_at(x-1, y):
            b[im] -= f_w_xm1_y*sim.w_bc_val(x-1, y)
        else:
            A[im, sim.w_nocc_idx(x-1, y)] = f_w_xm1_y
            eq_str += f"{f_w_xm1_y}*w_{x-1}{y} "

    if (y-1) >= 1:
        f_w_x_ym1 = -0.5*sim.alpha_y
        if sim.has_bc_at(x, y-1):
            b[im] -= f_w_x_ym1*sim.w_bc_val(x, y-1)
        else:
            A[im, sim.w_nocc_idx(x, y-1)] = f_w_x_ym1
            eq_str += f"{f_w_x_ym1}*w_{x}{y-1} "

    if ((x+1) <= sim.Nx) and ((y+1) <= sim.Ny):
        f_nonlin_1 = sim.beta/2

        if sim.has_bc_at(x, y+1) and sim.has_bc_at(x+1, y):
            b[im] -= f_nonlin_1*sim.w_bc_val(x+1, y)*sim.psi_bc_val(x, y+1)
        elif sim.has_bc_at(x, y+1):
            A[im, sim.w_nocc_idx(x+1, y)] = f_nonlin_1*sim.psi_bc_val(x, y+1)
            eq_str += f"{f_nonlin_1}*w_{x+1}{y} "
        elif sim.has_bc_at(x+1, y):
            A[im, sim.psi_nocc_idx(x, y+1)] = f_nonlin_1*sim.w_bc_val(x+1, y)
            eq_str += f"{f_nonlin_1}*p_{x}{y+1} "
        else:
            Q[im, sim.w_nocc_idx(x+1, y), sim.psi_nocc_idx(x, y+1)] = f_nonlin_1
            eq_str += f"{f_nonlin_1}*w_{x+1}{y}*p_{x}{y+1} "

    if (y+1) <= sim.Ny:
        f_nonlin_2 = -sim.beta/2
        if sim.has_bc_at(x, y+1) and sim.has_bc_at(x, y):
            b[im] -= f_nonlin_2*sim.w_bc_val(x, y)*sim.psi_bc_val(x, y+1)
        elif sim.has_bc_at(x, y+1):
            A[im, sim.w_nocc_idx(x, y)] = f_nonlin_2*sim.psi_bc_val(x, y+1)
            eq_str += f"{f_nonlin_2}*w_{x}{y} "
        elif sim.has_bc_at(x, y):
            A[im, sim.psi_nocc_idx(x, y+1)] = f_nonlin_2*sim.w_bc_val(x, y)
            eq_str += f"{f_nonlin_2}*p_{x}{y+1} "
        else:
            Q[im, sim.w_nocc_idx(x, y), sim.psi_nocc_idx(x, y+1)] = f_nonlin_2
            eq_str += f"{f_nonlin_2}*w_{x}{y}*p_{x}{y+1} "

    if (x+1) <= sim.Nx:
        f_nonlin_3 = -sim.beta/2
        if sim.has_bc_at(x, y) and sim.has_bc_at(x+1, y):
            b[im] -= f_nonlin_3*sim.w_bc_val(x+1, y)*sim.psi_bc_val(x, y)
        elif sim.has_bc_at(x, y):
            A[im, sim.w_nocc_idx(x+1, y)] = f_nonlin_3*sim.psi_bc_val(x, y)
            eq_str += f"{f_nonlin_3}*w_{x+1}{y} "
        elif sim.has_bc_at(x+1, y):
            A[im, sim.psi_nocc_idx(x, y)] = f_nonlin_3*sim.w_bc_val(x+1, y)
            eq_str += f"{f_nonlin_3}*p_{x}{y} "
        else:
            Q[im, sim.w_nocc_idx(x+1, y), sim.psi_nocc_idx(x, y)] = f_nonlin_3
            eq_str += f"{f_nonlin_3}*w_{x+1}{y}*p_{x}{y} "

    if ((x+1) <= sim.Nx) and ((y+1) <= sim.Ny):
        f_nonlin_4 = -sim.beta/2
        if sim.has_bc_at(x+1, y) and sim.has_bc_at(x, y+1):
            b[im] -= f_nonlin_4*sim.w_bc_val(x, y+1)*sim.psi_bc_val(x+1, y)
        elif sim.has_bc_at(x+1, y):
            A[im, sim.w_nocc_idx(x, y+1)] = f_nonlin_4*sim.psi_bc_val(x+1, y)
            eq_str += f"{f_nonlin_4}*w_{x}{y+1} "
        elif sim.has_bc_at(x, y+1):
            A[im, sim.psi_nocc_idx(x+1, y)] = f_nonlin_4*sim.w_bc_val(x, y+1)
            eq_str += f"{f_nonlin_4}*p_{x+1}{y} "
        else:
            Q[im, sim.w_nocc_idx(x, y+1), sim.psi_nocc_idx(x+1, y)] = f_nonlin_4
            eq_str += f"{f_nonlin_4}*w_{x}{y+1}*p_{x+1}{y} "

    if (y+1) <= sim.Ny:
        f_nonlin_5 = sim.beta/2
        if sim.has_bc_at(x, y) and sim.has_bc_at(x, y+1):
            b[im] -= f_nonlin_5*sim.w_bc_val(x, y+1)*sim.psi_bc_val(x, y)
        elif sim.has_bc_at(x, y):
            A[im, sim.w_nocc_idx(x, y+1)] = f_nonlin_5*sim.psi_bc_val(x, y)
            eq_str += f"{f_nonlin_5}*w_{x}{y+1} "
        elif sim.has_bc_at(x, y+1):
            A[im, sim.psi_nocc_idx(x, y)] = f_nonlin_5*sim.w_bc_val(x, y+1)
            eq_str += f"{f_nonlin_5}*p_{x}{y} "
        else:
            Q[im, sim.w_nocc_idx(x, y+1), sim.psi_nocc_idx(x, y)] = f_nonlin_5
            eq_str += f"{f_nonlin_5}*w_{x}{y+1}*p_{x}{y} "

    if (x+1) <= sim.Nx:
        f_nonlin_6 = sim.beta/2
        if sim.has_bc_at(x+1, y) and sim.has_bc_at(x, y):
            b[im] -= f_nonlin_6*sim.w_bc_val(x, y)*sim.psi_bc_val(x+1, y)
        elif sim.has_bc_at(x+1, y):
            A[im, sim.w_nocc_idx(x, y)] = f_nonlin_6*sim.psi_bc_val(x+1, y)
            eq_str += f"{f_nonlin_6}*w_{x}{y} "
        elif sim.has_bc_at(x, y):
            A[im, sim.psi_nocc_idx(x+1, y)] = f_nonlin_6*sim.w_bc_val(x, y)
            eq_str += f"{f_nonlin_6}*p_{x+1}{y} "
        else:
            Q[im, sim.w_nocc_idx(x, y), sim.psi_nocc_idx(x+1, y)] = f_nonlin_6
            eq_str += f"{f_nonlin_6}*w_{x}{y}*p_{x+1}{y} "

    """ This is how they must result
    Q[im, sim.w_idx(x+1, y), sim.psi_nocc_idx(x, y+1)] = sim.beta/2
    Q[im, sim.w_idx(x, y), sim.psi_nocc_idx(x, y+1)] = -sim.beta/2
    Q[im, sim.w_idx(x+1, y), sim.psi_nocc_idx(x, y)] = -sim.beta/2
    Q[im, sim.w_idx(x, y+1), sim.psi_nocc_idx(x+1, y)] = -sim.beta/2
    Q[im, sim.w_idx(x, y+1), sim.psi_nocc_idx(x, y)] = sim.beta/2
    Q[im, sim.w_idx(x, y), sim.psi_nocc_idx(x+1, y)] = sim.beta/2"""

    eq_str += f"= {b[im]}"
    if (sim.log & cfd.LOG_EQUATIONS) != 0: print(f"Transport[{im}] --- {eq_str}")

def calc_sim(sim: cfd.Simulation, quad_fn):
    qty_hist = []

    def has_psi_var(x, y):
        return (x >= 1) and (y >= 1) and (x <= sim.Nx) and (y <= sim.Ny) and not sim.has_bc_at(x, y)

    # Detect which equations can we use
    # (avoiding equations removed due to values fixed by BCs)
    tr_eqs = []
    for i in range(sim.Nx):
        for j in range(sim.Ny):
            x = i+1
            y = j+1
            tr_eqs.append((i,j))
    po_eqs = []
    for i in range(sim.Nx):
        for j in range(sim.Ny):
            x = i+1
            y = j+1
            if has_psi_var(x,y):
                po_eqs.append((i,j))

    if (sim.log & cfd.LOG_EQUATIONS) != 0: print(f"Ecuaciones: transporte={len(tr_eqs)}, poisson={len(po_eqs)}")

    # Preference order: tr, po
    EQ_TR = 1
    EQ_PO = 2
    eqs = []
    for eq in tr_eqs:
        eqs.append((EQ_TR, eq))
    for eq in po_eqs:
        eqs.append((EQ_PO, eq))

    assert len(eqs) >= sim.arr_size

    for t in range(sim.T):
        Q = np.zeros((sim.arr_size, sim.arr_size, sim.arr_size))
        A = np.zeros((sim.arr_size, sim.arr_size))
        b = np.zeros(sim.arr_size)

        # Prepare discretized equations
        for e in range(sim.arr_size):
            eq_type, eq = eqs[e]
            (eq_i, eq_j) = eq
            if eq_type == EQ_TR:
                apply_eq_transport_quad(sim, e, Q, A, b, eq_i+1, eq_j+1)
            elif eq_type == EQ_PO:
                # Es lineal
                cfdsim.lin_coupled.apply_eq_poisson(sim, e, A, b, eq_i+1, eq_j+1)

        np.set_printoptions(precision=3, suppress=True, edgeitems=10)
        if (sim.log & cfd.LOG_QUANTITIES) != 0:
            # Print Q, A, b
            print("Tensor Q:")
            print(Q)
            print("Matriz A:")
            print(A)
            print("\nVector b:")
            print(b)

        arr_sol = quad_fn(sim, sim.arr_size, Q, A, b)

        # Update current time-step with solution
        sim.update(arr_sol.copy())

    return qty_hist
