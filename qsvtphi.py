import pennylane as qml
from pyqsp import angle_sequence, poly
import numpy as np
import cacheutil

PHASES_DIR = "data-qsvt-phi"

"""def phase_path(kappa):
    if not os.path.isdir(PHASES_DIR):
        os.makedirs(PHASES_DIR)

    return f"{PHASES_DIR}/{int(np.ceil(kappa))}.csv"

def save_phases(kappa, phases):
    np.savetxt(phase_path(kappa), phases[:, None], delimiter=",")

def try_load_phases(kappa):
    if not os.path.isdir(PHASES_DIR):
        os.makedirs(PHASES_DIR)

    file = phase_path(kappa)
    if os.path.isfile(file):
        return np.loadtxt(file, delimiter=",")
    else:
        return None

def phases(kappa):
    load_phases = try_load_phases(kappa)
    if load_phases is not None:
        print(f"Loaded cache phases ({len(load_phases)}) for kappa={kappa}!")
        return (load_phases[0], load_phases[1:])
    else:
        print(f"Generating phases for kappa={kappa}...")

        # Generate QSVT phase sequence for c/x
        pcoefs, c = poly.PolyOneOverX().generate(kappa, return_coef=True, ensure_bounded=True, return_scale=True)
        c = c[0]
        
        phi_pyqsp = angle_sequence.QuantumSignalProcessingPhases(pcoefs, signal_operator="Wx", tolerance=1e-5)
        phiset = qml.transform_angles(phi_pyqsp, "QSP", "QSVT")

        phases = np.concatenate(([c], phiset))

        print(f"Caching phases ({len(phases)}, phases={phases})...")
        save_phases(kappa, phases)
        print(f"Done caching phases")
        return (c, phiset)"""

def phase_file(kappa):
    return f"{int(np.ceil(kappa))}.csv"

def save_phases(kappa, c_and_phases):
    cacheutil.save_data(PHASES_DIR, phase_file(kappa), c_and_phases)

def try_load_phases(kappa):
    return cacheutil.try_load_data(PHASES_DIR, phase_file(kappa))

def phases(kappa):
    load_phases = try_load_phases(kappa)
    if load_phases is not None:
        print(f"Loaded cache phases ({len(load_phases)}) for kappa={kappa}!")
        return (load_phases[0], load_phases[1:])
    else:
        print(f"Generating phases for kappa={kappa}...")

        # Generate QSVT phase sequence for c/x
        pcoefs, c = poly.PolyOneOverX().generate(kappa, return_coef=True, ensure_bounded=True, return_scale=True)
        c = c[0]
        
        phi_pyqsp = angle_sequence.QuantumSignalProcessingPhases(pcoefs, signal_operator="Wx", tolerance=1e-5)
        phiset = qml.transform_angles(phi_pyqsp, "QSP", "QSVT")

        phases = np.concatenate(([c], phiset))

        print(f"Caching phases ({len(phases)}, phases={phases})...")
        save_phases(kappa, phases)
        print(f"Done caching phases")
        return (c, phiset)
