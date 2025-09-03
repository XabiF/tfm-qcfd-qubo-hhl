import numpy as np
import cfd
import pennylane as qml
from pennylane.operation import Operation
from pyqsp import angle_sequence, poly
from scipy.linalg import svd, sqrtm
from .base import VAR_KIND_OMEGA, VAR_KIND_PSI, VAR_KIND_PSI_OMEGA
from qsvtphi import phases

# Custom PennyLane operator for us to supply an arbitrary matrix as a unitary

class CustomBE(Operation):
    num_params = 1
    num_wires = qml.operation.AnyWires
    grad_method = None

    def __init__(self, U, wires):
        super().__init__(U, wires=wires)

    @staticmethod
    def compute_matrix(*params):
        return params[0]

    def adjoint(self):
        UA = self.parameters[0]
        return CustomBE(qml.math.transpose(qml.math.conj(UA)), wires=self.wires)

    def label(self, decimals=None, base_label=None):
        return "UA"

def block_encode(q, n, A):
    U_svd, S, Vh = svd(A)
    alpha = np.linalg.norm(A, ord=np.inf)
    A_scaled = A / alpha

    m = 2**q - 2*n
    RT = sqrtm(np.eye(n) - (A_scaled @ A_scaled.conj().T))  # ensure it's PSD
    RB = sqrtm(np.eye(n) - (A_scaled.conj().T @ A_scaled))  # ensure it's PSD
    top = np.hstack((A_scaled, RT, np.zeros((n, m))))
    middle = np.hstack((RB, -(A_scaled).conj().T, np.zeros((n, m))))
    bottom = np.hstack((np.zeros((m, n)), np.zeros((m, n)), np.eye(m)))
    U = np.vstack((top, middle, bottom))  # 2**q x 2**q

    # Verify unitarity and block-encoding condition
    assert np.allclose(U.conj().T @ U, np.eye(2**q), atol=1e-5), "U not unitary"
    assert np.allclose(U[:n, :n], A_scaled, atol=1e-5), "Top-left not A/alpha"

    return U, alpha

def solve_hhl(sim: cfd.Simulation, n, A, b, kappa):
    # One extra qubit for the BE
    base_q = int(np.ceil(np.log2(n)))
    q = base_q + 1
    print(f"HHL qubit count: {q}")

    # Testing
    """_, sA, _ = svd(A)
    cond = max(sA) / min(sA)
    print(f"Condition number: {cond}")"""

    # Check for trivial solutions
    if np.allclose(b, 0):
        print("TRIVIAL!")
        return np.zeros(n)

    b_init = np.zeros(2**base_q, dtype=complex)
    b_init[:n] = b
    b_norm = np.linalg.norm(b_init)
    b_scaled = b_init / b_norm

    UA, alpha = block_encode(q, n, A)
    print(f"A norm: {alpha}")

    # Load QSVT phase sequence for c/x
    c, phiset = phases(kappa)

    print(f"QSVT gate count: {len(phiset)}")

    dev = qml.device("default.qubit", wires=q)
    @qml.qnode(dev)
    def qsvt_hhl():
        qml.StatePrep(b_scaled, wires=range(1,q))

        QUA = qml.QubitUnitary.compute_matrix(UA)
        UA_op = CustomBE(QUA, wires=range(q))

        qml.QSVT(
            UA_op,
            [qml.PCPhase(angle, dim=n, wires=range(q)) for angle in phiset]
        )
        return qml.state()
    
    denorm = b_norm / (alpha * c)

    # Run the circuit
    statevec = qsvt_hhl()
    sol_scaled = statevec[:n].real
    sol = sol_scaled * denorm

    # Testing (vs real solutions)
    """csol = np.linalg.solve(A, b)
    csol_scaled = csol / denorm"""

    """print(f"A: {A}")
    print(f"Anorm: {alpha}")
    print(f"b: {b}")
    print(f"bnorm: {b_norm}")
    print(f"denorm: {denorm}")
    print(f"csol_scaled: {csol_scaled}")
    print(f"qsol_scaled: {sol_scaled}")
    print(f"csol: {csol}")
    print(f"qsol: {sol}")"""

    return sol

def syslin_solve(sim: cfd.Simulation, n, A, b, var_kind):
    if var_kind == VAR_KIND_PSI_OMEGA:
        kappa = sim.param["hhl-kappa"]
    elif var_kind == VAR_KIND_OMEGA:
        kappa = sim.param["hhl-kappa-t"]
    elif var_kind == VAR_KIND_PSI:
        kappa = sim.param["hhl-kappa-p"]
    return solve_hhl(sim, n, A, b, kappa)
