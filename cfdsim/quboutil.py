import cfd
import numpy as np
from pyqubo import Array, Constraint
from dwave.samplers import SimulatedAnnealingSampler
from .quboenc import search_enc

# Utilities for QUBO resolution

def qubo_test_R(sim: cfd.Simulation):
    dpsi = sim.param["qubo-dyn-psi-base"]
    v1 = sim.param["qubo-dyn-intv-size"]
    v2 = 2-v1
    b = max(v1,v2)
    nu = sim.nu
    alphamin = 2
    alphamax = 3
    G = sim.Nx*sim.Ny
    N = int(np.sqrt(G))
    kappa = 4*dpsi*(b-1)
    R1 = int(np.log2(kappa/nu) + (alphamin/2)*np.log2(N) + np.log2(N-1))
    R2 = int(np.log2(kappa/nu) + (alphamax/2)*np.log2(N) + np.log2(N-1))
    print(f"Estimated R in [{min(R1,R2)},{max(R1,R2)}]")

def prepare_qubo_params(sim: cfd.Simulation, n, var_kind, encoding):
    qubo_test_R(sim)
    R = sim.param["qubo-R"]
    coefs_fn = search_enc(encoding)
    const_coefs = coefs_fn(sim, n, R, var_kind)
    # Binary encoding
    arr_bin = Array.create('arr', shape=(n, R), vartype='BINARY')
    if (sim.log & cfd.LOG_BASIC) != 0: print(f"Parameters, QUBO base: {n}*{R}={n*R}")
    arr_real = [(const_coefs[i][0] + sum((const_coefs[i][1][j]*arr_bin[i][j]) for j in range(R))) for i in range(n)]
    return (arr_real, const_coefs)

def process_qubo_params(sim: cfd.Simulation, n, const_coefs, qubo_sol):
    R = sim.param["qubo-R"]
    # Get real
    arr_sol = np.zeros(n)
    bit_tensor = np.zeros((n, R))
    for i in range(n):
        for j in range(R):
            pos_bit_name = f'arr[{i}][{j}]'
            pos_bit_val = qubo_sol[pos_bit_name]
            arr_sol[i] += const_coefs[i][1][j]*pos_bit_val
            bit_tensor[i, j] = pos_bit_val
        arr_sol[i] += const_coefs[i][0]
    return arr_sol, bit_tensor

def list_qubo_model_vars(model):
    qubo, offset = model.to_qubo()
    vs = []
    for k, v in qubo.items():
        v1, v2 = k
        if v1 not in vs:
            vs.append(v1)
        if v2 not in vs:
            vs.append(v2)
    return vs

def qubo_print_Q(bqm):
    n = len(bqm.variables)
    Q = np.zeros((n, n))
    variables = sorted(bqm.variables)
    var_index = {var: i for i, var in enumerate(variables)}
    for (var1, var2), coef in bqm.quadratic.items():
        i, j = var_index[var1], var_index[var2]
        Q[i, j] = coef

    diagonal_values = [
        bqm.linear[var] for var in variables
    ]
    np.fill_diagonal(Q, diagonal_values)

    maxQ = np.max(Q)
    minQ = np.min(Q)
    avgQ = np.mean(Q)
    print(f"[qubo] Q: max={maxQ:.4f}, min={minQ:.4f}, avg={avgQ:.4f}")

def qubo_solve(sim, objective):
    reads = sim.param["qubo-reads"]
    strength = sim.param["qubo-strength"]
    H = Constraint(objective, label="constraint")

    model = H.compile(strength=strength)
    vs = list_qubo_model_vars(model)
    if (sim.log & cfd.LOG_BASIC) != 0: print(f"Parameters, QUBO compiled: {len(vs)}")

    bqm = model.to_bqm()
    qubo_print_Q(bqm)

    # Solve QUBO via Simulated Annealing (offline)
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=reads)

    # Decode best solution
    decoded = model.decode_sampleset(sampleset)
    best = min(decoded, key=lambda x: x.energy)

    es_num = min(10, len(decoded))
    es = sorted([decoded[i].energy for i in range(es_num)])
    print(f"[qubo] Energies: {es}")

    solution = best.sample
    return solution
