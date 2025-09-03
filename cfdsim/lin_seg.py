import numpy as np
import cfd
import cfdplot
from .base import VAR_KIND_PSI, VAR_KIND_OMEGA
from .eqs import check_matriz_singular, analyze_spectrum, apply_eq_psi, apply_eq_omega

def calc_sim(sim: cfd.Simulation, lin_fn):
    qty_hist = []

    def has_var(x, y):
        return (x >= 1) and (y >= 1) and (x <= sim.Nx) and (y <= sim.Ny) and not sim.has_bc_at(x, y)

    # Detect which equations can we use
    # (avoiding equations removed due to values fixed by BCs)
    omega_eqs = []
    for i in range(sim.Nx):
        for j in range(sim.Ny):
            x = i+1
            y = j+1
            if has_var(x,y):
                omega_eqs.append((i,j))
    assert len(omega_eqs) >= sim.arr_w_count

    psi_eqs = []
    for i in range(sim.Nx):
        for j in range(sim.Ny):
            x = i+1
            y = j+1
            if has_var(x,y):
                psi_eqs.append((i,j))
    assert len(psi_eqs) >= sim.arr_psi_count

    if (sim.log & cfd.LOG_EQUATIONS) != 0: print(f"Equations: omega={len(omega_eqs)}, psi={len(psi_eqs)}")

    for t in range(sim.T):
        # Solve transport eq first
        Aw = np.zeros((sim.arr_w_count, sim.arr_w_count))
        bw = np.zeros(sim.arr_w_count)

        # Prepare discretized equations
        for e in range(sim.arr_w_count):
            (eq_i, eq_j) = omega_eqs[e]
            apply_eq_omega(sim, e, Aw, bw, eq_i+1, eq_j+1)

        np.set_printoptions(precision=3, suppress=True, edgeitems=10)
        if (sim.log & cfd.LOG_QUANTITIES) != 0:
            # Print A and b
            print("Matrix A (omega):")
            print(Aw)
            print("\nVector b (omega):")
            print(bw)
            cfdplot.plot_sparsity(Aw)
            analyze_spectrum(Aw, "Aw")

        # Check if our matrix was not correctly generated (mostly whether some columns or rows are all zero)
        # check_matriz_singular(Aw, sim.comps)

        temp_w_sol = lin_fn(sim, sim.arr_w_count, Aw, bw, VAR_KIND_OMEGA)

        # Intermediate update: fix omega BCs
        """base = sim.last_sol().copy()
        base[-len(temp_w_sol):] = temp_w_sol
        sim.arr_calc_w_bc(base)
        temp_w_sol = base[-len(temp_w_sol):]"""

        # Now we solve the Poisson eq

        Apsi = np.zeros((sim.arr_psi_count, sim.arr_psi_count))
        bpsi = np.zeros(sim.arr_psi_count)

        # Ecuaciones discretizadas, tantas como sean necesarias
        for e in range(sim.arr_psi_count):
            (eq_i, eq_j) = psi_eqs[e]
            apply_eq_psi(sim, e, Apsi, bpsi, eq_i+1, eq_j+1, temp_w_sol)

        np.set_printoptions(precision=3, suppress=True, edgeitems=10)
        if (sim.log & cfd.LOG_QUANTITIES) != 0:
            # Print A and b
            print("Matrix A (psi):")
            print(Apsi)
            print("\nVector b (psi):")
            print(bpsi)
            cfdplot.plot_sparsity(Apsi)
            analyze_spectrum(Apsi, "Apsi")

        # Check if our matrix was not correctly generated (mostly whether some columns or rows are all zero)
        # check_matriz_singular(Apsi, sim.comps)

        temp_psi_sol = lin_fn(sim, sim.arr_psi_count, Apsi, bpsi, VAR_KIND_PSI)

        #################################

        arr_sol = np.concatenate((temp_psi_sol, temp_w_sol))

        # Update current time-step with solution
        sim.update(arr_sol.copy())

    return qty_hist
