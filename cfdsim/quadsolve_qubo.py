import cfd
import cfdplot
from .quboutil import  prepare_qubo_params, process_qubo_params, qubo_solve

def sysquad_solve(sim: cfd.Simulation, n, Q, A, b):
    chimeraplot = sim.param["qubo-chimera-plot"]
    qubo_encoding = sim.param["qubo-encoding"]

    from .base import VAR_KIND_PSI_OMEGA

    # QUBO cost function: ||Q*x*x - A*x - b||^2
    arr_real, coefs = prepare_qubo_params(sim, n, None, VAR_KIND_PSI_OMEGA, qubo_encoding)
    objective = sum((sum(Q[i,j,k]*arr_real[j]*arr_real[k] for j in range(n) for k in range(n)) + sum(A[i,j]*arr_real[j] for j in range(n)) - b[i])**2 for i in range(n))

    solution = qubo_solve(sim, objective)
    arr_sol, bit_tensor = process_qubo_params(sim, n, coefs, solution)

    if chimeraplot:
        R = sim.param["R"]
        cfdplot.plot_qubo_chimera(n, R, bit_tensor)

    return arr_sol
