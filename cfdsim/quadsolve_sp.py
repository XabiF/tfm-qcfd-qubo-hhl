import numpy as np
import cfd
import sympy as sp

def sysquad_solve(sim: cfd.Simulation, n, Q, A, b):
    x_vec = sp.symbols(f'x0:{n}')

    # Use prev solutions as initial guesses
    psi_vec, w_vec, _, _ = sim.last_qty()
    x0 = np.zeros(n)
    for i in range(sim.Nx):
        for j in range(sim.Ny):
            x = i+1
            y = j+1
            if not sim.has_bc_at(x, y):
                x0[sim.psi_nocc_idx(x, y)] = psi_vec[i*sim.Ny + j]
                x0[sim.w_nocc_idx(x, y)] = w_vec[i*sim.Ny + j]

    eqs = []
    for i in range(n):
        eq_i = sum(Q[i,j,k]*x_vec[j]*x_vec[k] for j in range(n) for k in range(n)) + sum(A[i,j]*x_vec[j] for j in range(n)) - b[i]
        eqs.append(eq_i)

    # Solve numerically
    return np.array(list(sp.nsolve(eqs, x_vec, x0)))
