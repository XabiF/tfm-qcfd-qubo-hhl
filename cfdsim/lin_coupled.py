import numpy as np
import cfd
import cfdplot
from .base import VAR_KIND_PSI_OMEGA
from .eqs import check_matriz_singular, analyze_spectrum, apply_eq_poisson, apply_eq_transport

def calc_sim(sim: cfd.Simulation, lin_fn):
    qty_hist = []

    def has_var(x, y):
        return (x >= 1) and (y >= 1) and (x <= sim.Nx) and (y <= sim.Ny) and not sim.has_bc_at(x, y)

    # Detect which equations can we use
    # (avoiding equations removed due to values fixed by BCs)
    tr_eqs = []
    for i in range(sim.Nx):
        for j in range(sim.Ny):
            x = i+1
            y = j+1
            if has_var(x,y):
                tr_eqs.append((i,j))
    po_eqs = []
    for i in range(sim.Nx):
        for j in range(sim.Ny):
            x = i+1
            y = j+1
            if has_var(x,y):
                po_eqs.append((i,j))

    if (sim.log & cfd.LOG_EQUATIONS) != 0: print(f"Equations: transport={len(tr_eqs)}, poisson={len(po_eqs)}")

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
        # Create system matrix A and vector b
        A = np.zeros((sim.arr_size, sim.arr_size))
        b = np.zeros(sim.arr_size)

        # Prepare discretized equations
        for e in range(sim.arr_size):
            eq_type, eq = eqs[e]
            (eq_i, eq_j) = eq
            if eq_type == EQ_TR:
                apply_eq_transport(sim, e, A, b, eq_i+1, eq_j+1)
            elif eq_type == EQ_PO:
                apply_eq_poisson(sim, e, A, b, eq_i+1, eq_j+1)

        np.set_printoptions(precision=3, suppress=True, edgeitems=10)
        if (sim.log & cfd.LOG_QUANTITIES) != 0:
            # Print A and b
            print("Matrix A:")
            print(A)
            print("\nVector b:")
            print(b)
            cfdplot.plot_sparsity(A)
            analyze_spectrum(A)

        # Check if our matrix was not correctly generated (mostly whether some columns or rows are all zero)
        # check_matriz_singular(A, sim.comps)

        arr_sol = lin_fn(sim, sim.arr_size, A, b, VAR_KIND_PSI_OMEGA)

        # Update current time-step with solution
        sim.update(arr_sol.copy())

    return qty_hist
