import numpy as np
import cfd

def check_matriz_singular(A, arr_comps):
    zero_rows = np.where(~A.any(axis=1))[0]
    zero_row_terms = [arr_comps[x] for x in zero_rows]

    if (len(zero_rows) > 0) or (len(zero_row_terms) > 0):
        print(f"Zero rows: {zero_rows} | {zero_row_terms}")

    """repeated_rows = []
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            if np.allclose(A[i, :], A[j, :]):
                repeated_rows.append((i, j))
    if len(repeated_rows) > 0:
        print("Repeated rows:", repeated_rows)

    proportional_rows = []
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = A[i, :] / A[j, :]
                if np.allclose(ratio, ratio[0], equal_nan=True):
                    proportional_rows.append((i, j))
    if len(proportional_rows) > 0:
        print("Proportional rows:", proportional_rows)"""

    zero_cols = np.where(~A.any(axis=0))[0]
    zero_col_terms = [arr_comps[x] for x in zero_cols]
    if (len(zero_cols) > 0) or (len(zero_col_terms) > 0):
        print(f"Zero columns: {zero_cols} | {zero_col_terms}")

    """repeated_cols = []
    for i in range(A.shape[1]):
        for j in range(i + 1, A.shape[1]):
            if np.allclose(A[:, i], A[:, j]):
                repeated_cols.append((i, j))
    if len(repeated_cols) > 0:
        print("Repeated columns:", repeated_cols)

    proportional_cols = []
    for i in range(A.shape[1]):
        for j in range(i + 1, A.shape[1]):
            ratio = A[:, i] / A[:, j]
            if np.allclose(ratio, ratio[0], equal_nan=True):  # handle div by 0 or inf
                proportional_cols.append((i, j))
    if len(proportional_cols) > 0:
        print("Proportional columns:", proportional_cols)"""

def analyze_spectrum(A, name="A"):
    print(f"Analyzing spectral properties of {name}...\n")

    # Eigenvalues (only if A is diagonalizable)
    try:
        eigenvalues = np.linalg.eigvals(A)
        print(f"Eigenvalues of {name}:")
        print(np.round(eigenvalues, 4))
    except np.linalg.LinAlgError:
        print("Failed to compute eigenvalues (non-diagonalizable matrix).")

    # Singular values (always well-defined)
    singular_values = np.linalg.svd(A, compute_uv=False)
    print(f"\nSingular values of {name} (descending):")
    print(np.round(singular_values, 4))

    # Condition number
    cond = np.max(singular_values) / np.min(singular_values)
    print(f"\nCondition number (σ_max / σ_min): {cond:.4f}")
    print(f"\nInverse condition number (σ_min / σ_max): {(1/cond):.4f}")

def apply_eq_poisson(sim: cfd.Simulation, im, A, b, x, y):
    b[im] = 0
    eq_str = ""

    f_psi_x_y = -2*((1/(sim.dx**2)) + (1/(sim.dy**2)))
    if sim.has_bc_at(x, y):
        b[im] -= f_psi_x_y*sim.psi_bc_val(x, y)
    else:
        A[im, sim.psi_nocc_idx(x, y)] = f_psi_x_y
        eq_str += f"{f_psi_x_y}*p_{x}{y} "

    f_w_x_y = 1
    if sim.has_bc_at(x, y):
        b[im] -= f_w_x_y*sim.w_bc_val(x, y)
    else:
        A[im, sim.w_nocc_idx(x, y)] = f_w_x_y
        eq_str += f"{f_w_x_y}*w_{x}{y} "

    if (y+1) <= sim.Ny:
        f_psi_x_yp1 = 1/(sim.dy**2)
        if sim.has_bc_at(x, y+1):
            b[im] -= f_psi_x_yp1*sim.psi_bc_val(x, y+1)
        else:
            A[im, sim.psi_nocc_idx(x, y+1)] = f_psi_x_yp1
            eq_str += f"{f_psi_x_yp1}*p_{x}{y+1} "

    if (y-1) >= 1:
        f_psi_x_ym1 = 1/(sim.dy**2)
        if sim.has_bc_at(x, y-1):
            b[im] -= f_psi_x_ym1*sim.psi_bc_val(x, y-1)
        else:
            A[im, sim.psi_nocc_idx(x, y-1)] = f_psi_x_ym1
            eq_str += f"{f_psi_x_ym1}*p_{x}{y-1} "

    if (x+1) <= sim.Nx:
        f_psi_xp1_y = 1/(sim.dx**2)
        if sim.has_bc_at(x+1, y):
            b[im] -= f_psi_xp1_y*sim.psi_bc_val(x+1, y)
        else:
            A[im, sim.psi_nocc_idx(x+1, y)] = f_psi_xp1_y
            eq_str += f"{f_psi_xp1_y}*p_{x+1}{y} "

    if (x-1) >= 1:
        f_psi_xm1_y = 1/(sim.dx**2)
        if sim.has_bc_at(x-1, y):
            b[im] -= f_psi_xm1_y*sim.psi_bc_val(x-1, y)
        else:
            A[im, sim.psi_nocc_idx(x-1, y)] = f_psi_xm1_y
            eq_str += f"{f_psi_xm1_y}*p_{x-1}{y} "

    eq_str += f"= {b[im]}"
    if (sim.log & cfd.LOG_EQUATIONS) != 0: print(f"Poisson[{im}] --- {eq_str}")

def apply_eq_transport(sim: cfd.Simulation, im, A, b, x, y):
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

    # Crank-Nicolson, parte del instante anterior (constantes)
    F0 = 0
    F0 += sim.alpha_x*(arr_cur_w_xp1_y - 2*arr_cur_w_x_y + arr_cur_w_xm1_y)
    F0 += sim.alpha_y*(arr_cur_w_x_yp1 - 2*arr_cur_w_x_y + arr_cur_w_x_ym1)
    F0 -= sim.beta*((arr_cur_w_xp1_y - arr_cur_w_x_y)*(arr_cur_psi_x_yp1 - arr_cur_psi_x_y) - (arr_cur_w_x_yp1 - arr_cur_w_x_y)*(arr_cur_psi_xp1_y - arr_cur_psi_x_y))

    b[im] = arr_cur_w_x_y + 0.5*F0

    f_w_x_y = 1 + sim.alpha_x + sim.alpha_y + 0.25*sim.beta*(arr_cur_psi_xp1_y - arr_cur_psi_x_yp1)
    if sim.has_bc_at(x, y):
        b[im] -= f_w_x_y*sim.w_bc_val(x, y)
    else:
        A[im, sim.w_nocc_idx(x, y)] = f_w_x_y
        eq_str += f"{f_w_x_y}*w_{x}{y} "

    f_psi_x_y = -0.25*sim.beta*(arr_cur_w_xp1_y - arr_cur_w_x_yp1)
    if sim.has_bc_at(x, y):
        b[im] -= f_psi_x_y*sim.psi_bc_val(x, y)
    else:
        A[im, sim.psi_nocc_idx(x, y)] = f_psi_x_y
        eq_str += f"{f_psi_x_y}*p_{x}{y} "

    if (x+1) <= sim.Nx:
        f_w_xp1_y = -0.5*sim.alpha_x + 0.25*sim.beta*(arr_cur_psi_x_yp1 - arr_cur_psi_x_y)
        if sim.has_bc_at(x+1, y):
            b[im] -= f_w_xp1_y*sim.w_bc_val(x+1, y)
        else:
            A[im, sim.w_nocc_idx(x+1, y)] = f_w_xp1_y
            eq_str += f"{f_w_xp1_y}*w_{x+1}{y} "

        f_psi_xp1_y = -0.25*sim.beta*(arr_cur_w_x_yp1 - arr_cur_w_x_y)
        if sim.has_bc_at(x+1, y):
            b[im] -= f_psi_xp1_y*sim.psi_bc_val(x+1, y)
        else:
            A[im, sim.psi_nocc_idx(x+1, y)] = f_psi_xp1_y
            eq_str += f"{f_psi_xp1_y}*p_{x+1}{y} "

    if (y+1) <= sim.Ny:
        f_w_x_yp1 = -0.5*sim.alpha_y - 0.25*sim.beta*(arr_cur_psi_xp1_y - arr_cur_psi_x_y)
        if sim.has_bc_at(x, y+1):
            b[im] -= f_w_x_yp1*sim.w_bc_val(x, y+1)
        else:
            A[im, sim.w_nocc_idx(x, y+1)] = f_w_x_yp1
            eq_str += f"{f_w_x_yp1}*w_{x}{y+1} "

        f_psi_x_yp1 = 0.25*sim.beta*(arr_cur_w_xp1_y - arr_cur_w_x_y)
        if sim.has_bc_at(x, y+1):
            b[im] -= f_psi_x_yp1*sim.psi_bc_val(x, y+1)
        else:
            A[im, sim.psi_nocc_idx(x, y+1)] = f_psi_x_yp1
            eq_str += f"{f_psi_x_yp1}*p_{x}{y+1} "

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

    eq_str += f"= {b[im]}"
    if (sim.log & cfd.LOG_EQUATIONS) != 0: print(f"Transport[{im}] --- {eq_str}")

def apply_eq_psi(sim: cfd.Simulation, im, A, b, x, y, temp_w_vec):
    if sim.has_bc_at(x, y):
        b[im] = -sim.w_bc_val(x, y)
    else:
        b[im] = -temp_w_vec[sim.w_nocc_abs_idx(x, y)]
    eq_str = ""

    f_psi_x_y = -2*((1/(sim.dx**2)) + (1/(sim.dy**2)))
    if sim.has_bc_at(x, y):
        b[im] -= f_psi_x_y*sim.psi_bc_val(x, y)
    else:
        A[im, sim.psi_nocc_idx(x, y)] = f_psi_x_y
        eq_str += f"{f_psi_x_y}*p_{x}{y} "

    if (y+1) <= sim.Ny:
        f_psi_x_yp1 = 1/(sim.dy**2)
        if sim.has_bc_at(x, y+1):
            b[im] -= f_psi_x_yp1*sim.psi_bc_val(x, y+1)
        else:
            A[im, sim.psi_nocc_idx(x, y+1)] = f_psi_x_yp1
            eq_str += f"{f_psi_x_yp1}*p_{x}{y+1} "

    if (y-1) >= 1:
        f_psi_x_ym1 = 1/(sim.dy**2)
        if sim.has_bc_at(x, y-1):
            b[im] -= f_psi_x_ym1*sim.psi_bc_val(x, y-1)
        else:
            A[im, sim.psi_nocc_idx(x, y-1)] = f_psi_x_ym1
            eq_str += f"{f_psi_x_ym1}*p_{x}{y-1} "

    if (x+1) <= sim.Nx:
        f_psi_xp1_y = 1/(sim.dx**2)
        if sim.has_bc_at(x+1, y):
            b[im] -= f_psi_xp1_y*sim.psi_bc_val(x+1, y)
        else:
            A[im, sim.psi_nocc_idx(x+1, y)] = f_psi_xp1_y
            eq_str += f"{f_psi_xp1_y}*p_{x+1}{y} "

    if (x-1) >= 1:
        f_psi_xm1_y = 1/(sim.dx**2)
        if sim.has_bc_at(x-1, y):
            b[im] -= f_psi_xm1_y*sim.psi_bc_val(x-1, y)
        else:
            A[im, sim.psi_nocc_idx(x-1, y)] = f_psi_xm1_y
            eq_str += f"{f_psi_xm1_y}*p_{x-1}{y} "

    eq_str += f"= {b[im]}"
    if (sim.log & cfd.LOG_EQUATIONS) != 0: print(f"Psi[{im}] --- {eq_str}")

def apply_eq_omega(sim: cfd.Simulation, im, A, b, x, y):
    eq_str = ""

    u_x_y = sim.last_u_at(x, y)
    v_x_y = sim.last_v_at(x, y)

    w0_x_y = sim.arr_cur_w_at(x, y)
    w0_xp1_y = sim.arr_cur_w_at(x+1, y)
    if w0_xp1_y is None: w0_xp1_y = 0
    w0_xm1_y = sim.arr_cur_w_at(x-1, y)
    if w0_xm1_y is None: w0_xm1_y = 0
    w0_x_yp1 = sim.arr_cur_w_at(x, y+1)
    if w0_x_yp1 is None: w0_x_yp1 = 0
    w0_x_ym1 = sim.arr_cur_w_at(x, y-1)
    if w0_x_ym1 is None: w0_x_ym1 = 0

    F0 = 0
    F0 += sim.alpha_x*(w0_xp1_y -2*w0_x_y + w0_xm1_y)
    F0 += sim.alpha_y*(w0_x_yp1 -2*w0_x_y + w0_x_ym1)

    f_x_y = 1 + sim.alpha_x + sim.alpha_y
    if u_x_y > 0:
        f_x_y += (u_x_y/2)*(sim.dt/sim.dx)
        F0 += - u_x_y*(sim.dt/sim.dx)*(w0_x_y - w0_xm1_y)
    else:
        f_x_y -= (u_x_y/2)*(sim.dt/sim.dx)
        F0 += - u_x_y*(sim.dt/sim.dx)*(w0_xp1_y - w0_x_y)
    if v_x_y > 0:
        f_x_y += (v_x_y/2)*(sim.dt/sim.dx)
        F0 += - v_x_y*(sim.dt/sim.dy)*(w0_x_y - w0_x_ym1)
    else:
        f_x_y -= (v_x_y/2)*(sim.dt/sim.dx)
        F0 += - v_x_y*(sim.dt/sim.dy)*(w0_x_yp1 - w0_x_y)

    b[im] = w0_x_y + F0/2

    f_xp1_y = - sim.alpha_x/2
    if u_x_y < 0:
        f_xp1_y += (u_x_y/2)*(sim.dt/sim.dx)

    f_xm1_y = - sim.alpha_x/2
    if u_x_y > 0:
        f_xm1_y -= (u_x_y/2)*(sim.dt/sim.dx)

    f_x_yp1 = - sim.alpha_y/2
    if v_x_y < 0:
        f_x_yp1 += (v_x_y/2)*(sim.dt/sim.dx)

    f_x_ym1 = - sim.alpha_y/2
    if v_x_y > 0:
        f_x_ym1 -= (v_x_y/2)*(sim.dt/sim.dx)

    A[im, sim.w_nocc_abs_idx(x, y)] = f_x_y
    eq_str += f"{f_x_y}*p_{x}{y} "

    if (y+1) <= sim.Ny:
        if sim.has_bc_at(x, y+1):
            b[im] -= f_x_yp1*sim.w_bc_val(x, y+1)
        else:
            A[im, sim.w_nocc_abs_idx(x, y+1)] = f_x_yp1
            eq_str += f"{f_x_yp1}*p_{x}{y+1} "
    if (y-1) >= 1:
        if sim.has_bc_at(x, y-1):
            b[im] -= f_x_ym1*sim.w_bc_val(x, y-1)
        else:
            A[im, sim.w_nocc_abs_idx(x, y-1)] = f_x_ym1
            eq_str += f"{f_x_ym1}*p_{x}{y-1} "
    if (x+1) <= sim.Nx:
        if sim.has_bc_at(x+1, y):
            b[im] -= f_xp1_y*sim.w_bc_val(x+1, y)
        else:
            A[im, sim.w_nocc_abs_idx(x+1, y)] = f_xp1_y
            eq_str += f"{f_xp1_y}*p_{x+1}{y} "
    if (x-1) >= 1:
        if sim.has_bc_at(x-1, y):
            b[im] -= f_xm1_y*sim.w_bc_val(x-1, y)
        else:
            A[im, sim.w_nocc_abs_idx(x-1, y)] = f_xm1_y
            eq_str += f"{f_xm1_y}*p_{x-1}{y} "

    eq_str += f"= {b[im]}"
    if (sim.log & cfd.LOG_EQUATIONS) != 0: print(f"Omega[{im}] --- {eq_str}")
