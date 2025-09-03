import cfd
from .base import VAR_KIND_PSI_OMEGA, VAR_KIND_PSI, VAR_KIND_OMEGA

# Encoding techniques

##########################################################################33
# R/2 pos + R/2 neg

def qubo_encoding_pos_neg_half(sim: cfd.Simulation, n, R, var_kind):
    assert (R % 2) == 0
    j0 = sim.param["qubo-fixed-j0"]
    return [(0.0, [2**(j0-j) for j in range(R//2)] + [-2**(j0-j) for j in range(R//2)]) for i in range(n)]

##########################################################################
# R-1 pos + 1 neg

def qubo_encoding_pos_neg_1(sim: cfd.Simulation, n, R, var_kind):
    assert R > 1
    j0 = sim.param["qubo-fixed-j0"]
    return [(0.0, [2**(j0-j) for j in range(R-1)] + [-2**(j0+1)]) for i in range(n)]

##########################################################################
# "Dynamic"

def get_var_at(sim: cfd.Simulation, vec, is_psi, x, y):
    if (x < 1) or (y < 1) or (x > sim.Nx) or (y > sim.Ny):
        """if is_psi:
            return sim.calc_psi_bc(x, y)
        else:
            return None"""
        return None
    return vec[(x-1)*sim.Ny + (y-1)]

def is_zero(x):
    # Tune as needed
    return abs(x) < 1e-6

def qubo_encoding_dynamic(sim: cfd.Simulation, n, R, var_kind):
    (last_psi_vec, last_w_vec, _, _) = sim.last_qty()
    if sim.has_qty_prev():
        (prev_psi_vec, prev_w_vec, _, _) = sim.prev_qty()
    if sim.has_qty_prev2():
        (prev2_psi_vec, prev2_w_vec, _, _) = sim.prev2_qty()
    order = sim.param["qubo-dyn-t-order"]
    base_psi = sim.param["qubo-dyn-psi-base"]
    base_w = sim.param["qubo-dyn-w-base"]

    coefs = []
    for i in range(n):
        if ((var_kind == VAR_KIND_PSI_OMEGA) and (i < sim.arr_psi_count)) or (var_kind == VAR_KIND_PSI):
            # psi
            is_psi = True
            x, y = sim.nocc_xy(i)
                
            last_var_vec = last_psi_vec
            if sim.has_qty_prev():
                prev_var_vec = prev_psi_vec
            if sim.has_qty_prev2():
                prev2_var_vec = prev2_psi_vec
        elif ((var_kind == VAR_KIND_PSI_OMEGA) and (i >= sim.arr_psi_count)) or (var_kind == VAR_KIND_OMEGA):
            # w
            is_psi = False
            idx = i
            if (var_kind == VAR_KIND_PSI_OMEGA):
                idx = i - sim.arr_psi_count
            else:
                idx = i
            x, y = sim.nocc_xy(idx)

            last_var_vec = last_w_vec
            if sim.has_qty_prev():
                prev_var_vec = prev_w_vec
            if sim.has_qty_prev2():
                prev2_var_vec = prev2_w_vec
        else:
            raise RuntimeError("Invalid var_kind provided")

        var_x_y = last_var_vec[(x-1)*sim.Ny + (y-1)]

        if sim.has_qty_prev():
            diff_type = "t"
            # First-order Taylor on time derivative

            var_x_y_m1 = prev_var_vec[(x-1)*sim.Ny + (y-1)]

            if (order >= 2) and sim.has_qty_prev2():
                # Up to 2nd order
                var_x_y_m2 = prev2_var_vec[(x-1)*sim.Ny + (y-1)]
                diff = (3*var_x_y - 4*var_x_y_m1 + var_x_y_m2) / 2
            else:
                # Stay at 1st order
                diff = (var_x_y - var_x_y_m1)

        else:
            diff_type = "t0"
            # We have no prev instants to estimate time derivs
            # Alternative: get dw/dt from transport eq, then psi through specially-discretized poisson (or dt approx)

            w_xp1_y = get_var_at(sim, last_w_vec, False, x+1, y)
            if w_xp1_y is None: w_xp1_y = 0
            w_x_yp1 = get_var_at(sim, last_w_vec, False, x, y+1)
            if w_x_yp1 is None: w_x_yp1 = 0
            w_xm1_y = get_var_at(sim, last_w_vec, False, x-1, y)
            if w_xm1_y is None: w_xm1_y = 0
            w_x_ym1 = get_var_at(sim, last_w_vec, False, x, y-1)
            if w_x_ym1 is None: w_x_ym1 = 0
            psi_xp1_y = get_var_at(sim, last_psi_vec, True, x+1, y)
            if psi_xp1_y is None: psi_xp1_y = 0
            psi_x_yp1 = get_var_at(sim, last_psi_vec, True, x, y+1)
            if psi_x_yp1 is None: psi_x_yp1 = 0
            w_x_y = last_w_vec[(x-1)*sim.Ny + (y-1)]
            psi_x_y = last_psi_vec[(x-1)*sim.Ny + (y-1)]

            partialt_w_x_y = 0
            partialt_w_x_y += sim.nu*((w_xp1_y - 2*w_x_y + w_xm1_y) / (sim.dx**2))
            partialt_w_x_y += sim.nu*((w_x_yp1 - 2*w_x_y + w_x_ym1) / (sim.dy**2))
            partialt_w_x_y -= ((psi_x_yp1 - psi_x_y) / sim.dy) * ((w_xp1_y - w_x_y) / sim.dx)
            partialt_w_x_y += ((psi_xp1_y - psi_x_y) / sim.dx) * ((w_x_yp1 - w_x_y) / sim.dy)
            diffnext_w_x_y = partialt_w_x_y*sim.dt

            # print(f"{v}[{x}{y}] partialt_w_x_y={partialt_w_x_y}")

            if is_psi:
                w_x_y_next = w_x_y + diffnext_w_x_y
                
                psi_xm1_y = get_var_at(sim, last_psi_vec, True, x-1, y)
                if psi_xm1_y is None: psi_xm1_y = 0
                psi_x_ym1 = get_var_at(sim, last_psi_vec, True, x, y-1)
                if psi_x_ym1 is None: psi_x_ym1 = 0
                psi_x_y_next = (w_x_y_next + (psi_xp1_y + psi_xm1_y)/(sim.dx**2) + (psi_x_yp1 + psi_x_ym1)/(sim.dy**2)) / (2*((1/sim.dx**2) + (1/sim.dy**2)))
                diff = psi_x_y_next - psi_x_y

                """partialt_psi_x_y = - sim.nu * w_x_y_next
                diff = partialt_psi_x_y*sim.dt"""
            else:
                diff = diffnext_w_x_y

        v1 = sim.param["qubo-dyn-intv-size"]
        assert v1 <= 2
        assert v1 >= 0
        if is_zero(diff):
            if is_psi:
                base = base_psi
            else:
                base = base_w
            # zero diff: interval [-(b-1)*base, (b-1)*base] of size 2*(b-1)*base, centered at 0
            b = max(v1, 2-v1)
            limits = [-(b-1)*base, (b-1)*base]
        else:
            # non-zero diff: interval [a*diff, b*diff] of size 2*(b-1)*diff, centered at diff
            limits = [v1*diff, (2-v1)*diff]
        
        max_diff = max(limits)
        min_diff = min(limits)
    
        intv_diff = max_diff - min_diff
        const = var_x_y + min_diff + intv_diff*2**(-(R+1))
        # print(f"{v}[{x}{y}] prev={var_x_y:.3f} ---{diff_type}--- min={min_diff}, max={max_diff}, intvlen={max_diff-min_diff}")
        coefs_i = [intv_diff*(2**(-j)) for j in range(1,R+1)]
        assert len(coefs_i) == R

        coefs.append((const, coefs_i))
        # print(f"{v}[{x},{y}] coefs finales: {coefs_i}")

    return coefs

def search_enc(name):
    if name == "pos-neg-half":
        return qubo_encoding_pos_neg_half
    elif name == "pos-neg-1":
        return qubo_encoding_pos_neg_1
    elif name == "dyn":
        return qubo_encoding_dynamic
    else:
        raise ValueError(f"Bad encoding '{name}'")
