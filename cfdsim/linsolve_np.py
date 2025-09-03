import numpy as np
import cfd

def syslin_solve(sim: cfd.Simulation, n, A, b, var_kind):
    return np.linalg.solve(A, b)
