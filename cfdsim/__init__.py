from .lin_coupled import calc_sim as calc_sim_lin_coupled
from .lin_seg import calc_sim as calc_sim_lin_seg
from .linsolve_np import syslin_solve as syslin_solver_np
from .linsolve_qubo import syslin_solve as syslin_solver_qubo
from .linsolve_hhl import syslin_solve as syslin_solver_hhl

from .quad import calc_sim as calc_sim_quad
from .quadsolve_sp import sysquad_solve as sysquad_solver_sp
from .quadsolve_qubo import sysquad_solve as sysquad_solver_qubo

from .quboenc import qubo_encoding_pos_neg_half
from .quboenc import qubo_encoding_pos_neg_1
from .quboenc import qubo_encoding_dynamic

import cfd

def perform_sim(sim: cfd.Simulation):
    if "mode" not in sim.param:
        raise ValueError(f"No mode was provided")

    sim_mode = sim.param["mode"]

    # Old linearization approach, use segregated instead!
    if sim_mode == "lin-cpl-np":
        calc_sim_lin_coupled(sim, syslin_solver_np)
    elif sim_mode == "lin-cpl-qubo":
        calc_sim_lin_coupled(sim, syslin_solver_qubo)
    elif sim_mode == "lin-cpl-hhl":
        calc_sim_lin_coupled(sim, syslin_solver_hhl)

    elif sim_mode == "lin-seg-np":
        calc_sim_lin_seg(sim, syslin_solver_np)
    elif sim_mode == "lin-seg-qubo":
        calc_sim_lin_seg(sim, syslin_solver_qubo)
    elif sim_mode == "lin-seg-hhl":
        calc_sim_lin_seg(sim, syslin_solver_hhl)

    elif sim_mode == "quad-sp":
        calc_sim_quad(sim, sysquad_solver_sp)
    elif sim_mode == "quad-qubo":
        calc_sim_quad(sim, sysquad_solver_qubo)

    else:
        raise ValueError(f"Unknown mode '{sim_mode}'")

def format_sim(sim: cfd.Simulation, sep=", ", others={}):
    params = sim.param.copy() | others

    # Don't format internal parameters
    del params["log"]
    del params["qubo-chimera-plot"]

    return sep.join(f'{k}={v}' for k, v in params.items())
