import numpy as np
import cacheutil

# bitflags
LOG_BASIC = 1<<0
LOG_VARIABLES_BCS = 1<<1
LOG_EQUATIONS = 1<<2
LOG_QUANTITIES = 1<<3
LOG_SOLUTIONS = 1<<4

SIM_DIR = "data-cfd-sim"

class Simulation:
    def __init__(self, param):
        self.param = param # Parameters (general + extra from solvers and so on)
        self.log = param["log"] # Print settings

        if "icbc-fns" in param:
            self.uv_ic, self.uv_bc, self.psi_0, self.char = param["icbc-fns"] # IC+BC directly provided as functions
        elif "icbc" in param:
            from cfdcc import search_ic_bc
            self.uv_ic, self.uv_bc, self.psi_0, self.char = search_ic_bc(param["icbc"]) # Built-in kinds of IC/BCs
        else:
            raise ValueError("No IC/BC provided!")

    ##############################################################
    def has_bc_at(self, x, y):
        return self.uv_bc(self, x, y) is not None

    def psi_0_val(self, x, y):
        return self.psi_0(self, x, y)

    def psi_bc_val(self, x, y):
        return self.psi_bc_map[self.psi_bc_idx(x, y)]
    
    def w_0_val(self, x, y):
        (u_x_y, v_x_y) = self.uv_ic(self, x, y)

        dv_dx = 0
        if ((x-1) >= 1) and ((x+1) <= self.Nx):
            (_, v_xm1) = self.uv_ic(self, x-1, y)
            (_, v_xp1) = self.uv_ic(self, x+1, y)
            dv_dx = (v_xp1 - v_xm1) / (2*self.dx)
        elif ((x-1) >= 1):
            (_, v_xm1) = self.uv_ic(self, x-1, y)
            dv_dx = (v_x_y - v_xm1) / (self.dx)
        elif ((x+1) <= self.Nx):
            (_, v_xp1) = self.uv_ic(self, x+1, y)
            dv_dx = (v_xp1 - v_x_y) / (self.dx)
        else:
            raise RuntimeError("nonsense")

        du_dy = 0
        if ((y-1) >= 1) and ((y+1) <= self.Ny):
            (u_ym1, _) = self.uv_ic(self, x, y-1)
            (u_yp1, _) = self.uv_ic(self, x, y+1)
            du_dy = (u_yp1 - u_ym1) / (2*self.dy)
        elif ((y-1) >= 1):
            (u_ym1, _) = self.uv_ic(self, x, y-1)
            du_dy = (u_x_y - u_ym1) / (self.dy)
        elif ((y+1) <= self.Ny):
            (u_yp1, _) = self.uv_ic(self, x, y+1)
            du_dy = (u_yp1 - u_x_y) / (self.dy)
        else:
            raise RuntimeError("nonsense")

        w_x_y = dv_dx - du_dy
        return w_x_y
    
    def w_bc_val(self, x, y):
        return self.w_bc_map[self.w_bc_idx(x, y)]
    
    def nocc_xy(self, arr_psi_idx):
        return self.reverse_nocc_map[arr_psi_idx]
        
    def psi_nocc_idx(self, x, y):
        return self.nocc_map[(x, y)]
    
    def psi_bc_idx(self, x, y):
        return self.cc_map[(x, y)]
    
    def psi_nocc_abs_idx(self, x, y):
        return self.nocc_map[(x, y)]
    
    def w_nocc_idx(self, x, y):
        return self.arr_psi_count + self.nocc_map[(x, y)]
    
    def w_bc_idx(self, x, y):
        return self.cc_map[(x, y)]
    
    def w_nocc_abs_idx(self, x, y):
        return self.nocc_map[(x, y)]

    def arr_psi_at_nocheck(self, arr, x, y):
        return arr[self.psi_nocc_idx(x, y)]
    
    def arr_w_at_nocheck(self, arr, x, y):
        return arr[self.w_nocc_idx(x, y)]
    
    def arr_psi_at(self, arr, x, y):
        if (x < 1) or (x > self.Nx) or (y < 1) or (y > self.Ny):
            return None

        if self.has_bc_at(x, y):
            return self.psi_bc_val(x, y)
        else:
            return self.arr_psi_at_nocheck(arr, x, y)

    def arr_w_at(self, arr, x, y):
        if (x < 1) or (x > self.Nx) or (y < 1) or (y > self.Ny):
            return None

        if self.has_bc_at(x, y):
            return self.w_bc_val(x, y)
        else:
            return self.arr_w_at_nocheck(arr, x, y)

    def arr_cur_psi_at(self, x, y):
        return self.arr_psi_at(self.arr_cur, x, y)
    
    def arr_cur_w_at(self, x, y):
        return self.arr_w_at(self.arr_cur, x, y)

    def last_u_at(self, x, y):
        return self.qty_cur[2][(x-1)*self.Ny + (y-1)]
    
    def last_v_at(self, x, y):
        return self.qty_cur[3][(x-1)*self.Ny + (y-1)]
    
    def calc_psi_0(self):
        for i in range(self.Nx):
            for j in range(self.Ny):
                x = i+1
                y = j+1
                psi_x_y = self.psi_0_val(x, y)

                if not self.has_bc_at(x, y):
                    self.arr_cur[self.psi_nocc_idx(x, y)] = psi_x_y
                else:
                    self.psi_bc_map[self.psi_bc_idx(x, y)] = psi_x_y

                if (self.log & LOG_VARIABLES_BCS) != 0: print(f"psi0[{x},{y}] = {psi_x_y})")

    def calc_w_0(self):
        for i in range(self.Nx):
            for j in range(self.Ny):
                x = i+1
                y = j+1
                w_x_y = self.w_0_val(x, y)

                if not self.has_bc_at(x, y):
                    self.arr_cur[self.w_nocc_idx(x, y)] = w_x_y
                else:
                    self.w_bc_map[self.w_bc_idx(x, y)] = w_x_y

                if (self.log & LOG_VARIABLES_BCS) != 0: print(f"w0[{x},{y}] = {w_x_y})")

    def calc_psi_bc(self):
        for (x, y), psi_idx in self.cc_map.items():
            (u_bc, v_bc) = self.uv_bc(self, x, y)

            # Locate neighboring point that exists and is not fixed by BCs
            if ((x-1) >= 1) and self.has_bc_at(x-1, y):
                # top/bottom
                psi_xm1_y = self.psi_bc_val(x-1, y)
                psi_x_y = psi_xm1_y - v_bc*self.dx
            elif ((x+1) <= self.Nx) and self.has_bc_at(x+1, y):
                # top/bottom
                psi_xp1_y = self.psi_bc_val(x+1, y)
                psi_x_y = psi_xp1_y + v_bc*self.dx
            elif ((y-1) >= 1 and self.has_bc_at(x, y-1)):
                # left/right
                psi_x_ym1 = self.psi_bc_val(x, y-1)
                psi_x_y = psi_x_ym1 + u_bc*self.dy
            elif ((y+1) <= self.Ny) and self.has_bc_at(x, y+1):
                # left/right
                psi_x_yp1 = self.psi_bc_val(x, y+1)
                psi_x_y = psi_x_yp1 - u_bc*self.dy
            else:
                psi_x_y = self.psi_0_val(x, y)

            self.psi_bc_map[self.psi_bc_idx(x, y)] = psi_x_y
            if (self.log & LOG_VARIABLES_BCS) != 0: print(f"psicc[{x},{y}] = {psi_x_y})")

    def w_term_bc_withxp1(self, x, y, cc_idx):
        (_, v_bc) = self.uv_bc(self, x, y)
        psi_x_y = self.psi_bc_map[cc_idx]
        psi_xp1_y = self.arr_cur_psi_at(x+1, y)
        return - (2*psi_xp1_y + 2*self.dx*v_bc - 2*psi_x_y) / self.dx**2
    
    def w_term_bc_withxm1(self, x, y, cc_idx):
        (_, v_bc) = self.uv_bc(self, x, y)
        psi_x_y = self.psi_bc_map[cc_idx]
        psi_xm1_y = self.arr_cur_psi_at(x-1, y)
        return - (2*psi_xm1_y - 2*self.dx*v_bc - 2*psi_x_y) / self.dx**2
    
    def w_term_bc_withyp1(self, x, y, cc_idx):
        (u_bc, _) = self.uv_bc(self, x, y)
        psi_x_y = self.psi_bc_map[cc_idx]
        psi_x_yp1 = self.arr_cur_psi_at(x, y+1)
        return - (2*psi_x_yp1 - 2*self.dy*u_bc - 2*psi_x_y) / self.dy**2
    
    def w_term_bc_withym1(self, x, y, cc_idx):
        (u_bc, _) = self.uv_bc(self, x, y)
        psi_x_y = self.psi_bc_map[cc_idx]
        psi_x_ym1 = self.arr_cur_psi_at(x, y-1)
        return - (2*psi_x_ym1 + 2*self.dy*u_bc - 2*psi_x_y) / self.dy**2

    def calc_w_bc(self):
        for (x, y), cc_idx in self.cc_map.items():
            # Locate neighboring point that exists and is not fixed by BCs
            if ((x-1) >= 1) and not self.has_bc_at(x-1, y):
                # include x-1
                w_x_y = self.w_term_bc_withxm1(x, y, cc_idx)
                # check if y+1 o y-1 do not exist or are BC points
                if ((y-1) < 1) or self.has_bc_at(x, y-1):
                    # we must use y+1
                    w_x_y += self.w_term_bc_withyp1(x, y, cc_idx)
                else:
                    w_x_y += self.w_term_bc_withym1(x, y, cc_idx)
            elif ((x+1) <= self.Nx) and not self.has_bc_at(x+1, y):
                # include x+1
                w_x_y = self.w_term_bc_withxp1(x, y, cc_idx)
                # check if y+1 o y-1 do not exist or are BC points
                if ((y-1) < 1) or self.has_bc_at(x, y-1):
                    # we must use y+1
                    w_x_y += self.w_term_bc_withyp1(x, y, cc_idx)
                else:
                    w_x_y += self.w_term_bc_withym1(x, y, cc_idx)
            elif ((y-1) >= 1) and not self.has_bc_at(x, y-1):
                # include y-1
                w_x_y = self.w_term_bc_withym1(x, y, cc_idx)
                # check if x+1 o x-1 do not exist or are BC points
                if ((x-1) < 1) or self.has_bc_at(x-1, y):
                    # we must use y+1
                    w_x_y += self.w_term_bc_withxp1(x, y, cc_idx)
                else:
                    w_x_y += self.w_term_bc_withxm1(x, y, cc_idx)
            elif ((y+1) <= self.Ny) and not self.has_bc_at(x, y+1):
                # include y+1
                w_x_y = self.w_term_bc_withyp1(x, y, cc_idx)
                # check if x+1 o x-1 do not exist or are BC points
                if ((x-1) < 1) or self.has_bc_at(x-1, y):
                    # we must use y+1
                    w_x_y += self.w_term_bc_withxp1(x, y, cc_idx)
                else:
                    w_x_y += self.w_term_bc_withxm1(x, y, cc_idx)
            else:
                # remaining cases, compute them based on neighboring BCs
                w_x_y = self.w_0_val(x, y)

            self.w_bc_map[cc_idx] = w_x_y
            if (self.log & LOG_VARIABLES_BCS) != 0: print(f"wcc[{x},{y}] = {w_x_y})")
        
    # Includes BCs for psi and returns a complete Nx*Ny vector
    def calc_psi_vec(self, arr):
        psi_new = np.zeros(self.Nx*self.Ny)
        for i in range(self.Nx):
            for j in range(self.Ny):
                x = i+1
                y = j+1
                psi_new[i*self.Ny + j] = self.arr_psi_at(arr, x, y)
        return psi_new

    # Includes BCs for w and returns a complete Nx*Ny vector
    def calc_w_vec(self, arr):
        w_new = np.zeros(self.Nx*self.Ny)
        for i in range(self.Nx):
            for j in range(self.Ny):
                x = i+1
                y = j+1
                w_new[i*self.Ny + j] = self.arr_w_at(arr, x, y)
        return w_new

    # Computes u,v from psi
    def calc_u(self, psi_vec):
        u_vec = np.zeros(self.Nx*self.Ny)
        v_vec = np.zeros(self.Nx*self.Ny)
        for i in range(self.Nx):
            for j in range(self.Ny):
                x = i+1
                y = j+1

                if self.has_bc_at(x, y):
                    (u_i_j, v_i_j) = self.uv_bc(self, x, y)
                else:
                    psi_i_j = psi_vec[i*self.Ny + j]

                    if ((x-1) >= 1) and ((x+1) <= self.Nx):
                        psi_ip1_j = psi_vec[(i+1)*self.Ny + j]
                        psi_im1_j = psi_vec[(i-1)*self.Ny + j]
                        v_i_j = - (psi_ip1_j - psi_im1_j) / (2*self.dx)
                    elif ((x-1) >= 1):
                        psi_im1_j = psi_vec[(i-1)*self.Ny + j]
                        v_i_j = - (psi_i_j - psi_im1_j) / self.dx
                    elif ((x+1) <= self.Nx):
                        psi_ip1_j = psi_vec[(i+1)*self.Ny + j]
                        v_i_j = - (psi_ip1_j - psi_i_j) / self.dx

                    if ((y-1) >= 1) and ((y+1) <= self.Ny):
                        psi_i_jp1 = psi_vec[i*self.Ny + (j+1)]
                        psi_i_jm1 = psi_vec[i*self.Ny + (j-1)]
                        u_i_j = (psi_i_jp1 - psi_i_jm1) / (2*self.dy)
                    elif ((y-1) >= 1):
                        psi_i_jm1 = psi_vec[i*self.Ny + (j-1)]
                        u_i_j = (psi_i_j - psi_i_jm1) / self.dy
                    elif ((y+1) <= self.Ny):
                        psi_i_jp1 = psi_vec[i*self.Ny + (j+1)]
                        u_i_j = (psi_i_jp1 - psi_i_j) / self.dy

                u_vec[i*self.Ny + j] = u_i_j
                v_vec[i*self.Ny + j] = v_i_j
        return (u_vec, v_vec)

    # Make a tuple of all quantities
    def calc_qties(self, arr):
        psi_vec = self.calc_psi_vec(arr)
        w_vec = self.calc_w_vec(arr)
        u_vec, v_vec = self.calc_u(psi_vec)
        return (psi_vec, w_vec, u_vec, v_vec)

    def prepare(self):
        # For simple square grids
        if "N" in self.param:
            self.param["Nx"] = self.param["N"]
            self.param["Ny"] = self.param["N"]
        if "L" in self.param:
            self.param["Lx"] = self.param["L"]
            self.param["Ly"] = self.param["L"]

        self.Nx = self.param["Nx"] # X and Y grid points
        self.Ny = self.param["Ny"]

        self.T = self.param["T"] # Time-step count / final time-step

        self.Lx = self.param["Lx"] # Grid dimensions for X and Y
        self.Ly = self.param["Ly"]

        self.dx = self.Lx / (self.Nx - 1)
        self.dy = self.Ly / (self.Ny - 1)

        self.dt = self.param["dt"] # Time increment

        if "Re" in self.param:
            has_re = True
            self.re = self.param["Re"] # Reynolds number
        elif "nu" in self.param:
            has_re = False
            self.nu = self.param["nu"] # Kinematic viscosity
        else:
            raise RuntimeError("No nu or Re was provided!")

        self.rho = self.param["rho"] # Density

        self.name = f"cfdsim-{self.Nx}x{self.Ny}-{self.param["icbc"]}-{self.param["mode"]}"

        self.cc_map = {}
        self.nocc_map = {}
        self.reverse_bc_map = {}
        self.reverse_nocc_map = {}
        cc_count = 0
        nocc_count = 0
        for i in range(self.Nx):
            for j in range(self.Ny):
                x = i+1
                y = j+1
                if self.has_bc_at(x, y):
                    if (self.log & LOG_VARIABLES_BCS) != 0: print(f"cc[{x},{y}]!")
                    self.cc_map[(x, y)] = cc_count
                    self.reverse_bc_map[cc_count] = (x, y)
                    cc_count += 1
                else:
                    self.nocc_map[(x, y)] = nocc_count
                    self.reverse_nocc_map[nocc_count] = (x, y)
                    nocc_count += 1

        self.psi_bc_map = np.zeros(cc_count)
        self.w_bc_map = np.zeros(cc_count)

        # We only need to solve non-fixed points!
        self.cc_comp_count = cc_count # number of variables (psi & omega per point) fixed by BCs
        self.arr_psi_count = self.Nx*self.Ny - self.cc_comp_count # psi components (system variables) to solve
        self.arr_w_count = self.Nx*self.Ny - self.cc_comp_count # w components (system variables) to solve
        self.arr_size = self.arr_psi_count + self.arr_w_count # total system variables

        # first "arr_psi_count" values are psi, next "arr_w_count" ones are w
        self.arr_cur = np.zeros(self.arr_size)

        # Compute ICs within current solution vector
        self.calc_psi_0()
        self.calc_w_0()

        self.comps = [""] * self.arr_size
        for i in range(self.arr_size):
            if i >= self.arr_psi_count:
                iw = i - self.arr_psi_count
                x,y = self.nocc_xy(iw)
                self.comps[i] = f"w_{x}{y}"
            else:
                x,y = self.nocc_xy(i)
                self.comps[i] = f"psi_{x}{y}"
        if (self.log & LOG_VARIABLES_BCS) != 0: print(f"To solve: ({", ".join(self.comps)})")

        self.cur_step = 0

        self.sol_prev = None
        self.sol_prev2 = None
        self.qty_prev = None
        self.qty_prev2 = None

        # Initial state data
        self.sol_cur = self.arr_cur.copy()
        self.qty_cur = self.calc_qties(self.sol_cur)

        cacheutil.save_data(self.sim_step_data_dir(), self.sim_step_data_name(self.cur_step), self.sol_cur)

        # Characteristic problem velocity and length
        uc, lc = self.char(self)

        # Reynolds number/viscosity estimation
        if has_re:
            self.nu = (uc*lc)/self.re
            if (self.log & LOG_BASIC) != 0: print(f"Viscosity: nu={self.nu}, Re={self.re} (provided Re)")
        else:
            self.re = int(np.ceil((uc*lc)/self.nu))
            if (self.log & LOG_BASIC) != 0: print(f"Viscosity: nu={self.nu}, Re={self.re} (provided nu)")

        # Discretized equation constants
        self.alpha_x = (self.nu * self.dt) / (self.dx**2)
        self.alpha_y = (self.nu * self.dt) / (self.dy**2)
        self.beta = self.dt / (self.dx*self.dy)
        if (self.log & LOG_BASIC) != 0: print(f"Constants: alphax={self.alpha_x:.5f}, alphay={self.alpha_y:.5f}, beta={self.beta:.5f}")

        # Pointless check since we employ Crank-Nicolson
        """crit = alpha_x + alpha_y <= 0.5
        if crit:
            print(f"Stability ok, alphax+alphay={alpha_x+alpha_y} <= 0.5")
        else:
            raise RuntimeError(f"Bad stability! alphax+alphay={alpha_x+alpha_y}")
            pass"""

        if (self.log & LOG_BASIC) != 0:
            print(f"Re={self.re}")

            # Gross estimation of required gridpoints to resolve turbulences in DNS
            G = self.re**3
            Nest = int(np.ceil(np.sqrt(G)))
            Nprom = int(np.ceil(np.sqrt(self.Nx*self.Ny)))
            if Nest > Nprom:
                print(f"Estimated DNS required gridpoints: N={Nest}>{Nprom} used! Potential instability...")
            elif Nest < Nprom:
                print(f"Estimated DNS required gridpoints: N={Nest}<{Nprom} used (good)")

            print(f"Total components (psi+w): {2*self.Nx*self.Ny}")
            print(f"Total variables: {self.arr_size}")
            print(f"Variables (psi): {self.arr_psi_count}")
            print(f"Variables (w): {self.arr_w_count}")
            print(f"BC points psi/w: {self.cc_comp_count}")

    def sim_step_data_dir(self):
        return f"{SIM_DIR}/{self.name}"
    
    def sim_step_data_name(self, i):
        return f"{i}.csv"

    def update(self, arr):
        self.cur_step += 1

        # Fix BC in w & psi
        self.calc_psi_bc()
        self.calc_w_bc()

        self.sol_prev2 = self.sol_prev
        self.sol_prev = self.sol_cur
        self.arr_cur = arr
        self.sol_cur = self.arr_cur.copy()

        self.qty_prev2 = self.qty_prev
        self.qty_prev = self.qty_cur
        self.qty_cur = self.calc_qties(self.sol_cur)
        cacheutil.save_data(self.sim_step_data_dir(), self.sim_step_data_name(self.cur_step), self.sol_cur)

        # CFL-like values (orientative)
        (psi, w, u, v) = self.qty_cur
        maxu = max(abs(max(u)), abs(min(u)))
        maxv = max(abs(max(v)), abs(min(v)))
        maxcx = maxu*(self.dt/self.dx)
        maxcy = maxv*(self.dt/self.dy)
        print(f"C({self.cur_step}/{self.T}): cx={maxcx:.3f}, cy={maxcy:.3f}, maxu={maxu}, maxv={maxv}")

        # Testing for tgv 10x10
        point = (4, 4)
        u_p = u[(point[0]-1)*self.Ny + (point[1]-1)]
        v_p = v[(point[0]-1)*self.Ny + (point[1]-1)]
        print(f"u_p={u_p}, v_p={v_p}")

        if (self.log & LOG_SOLUTIONS) != 0:
            print(f"Sol({self.cur_step}/{self.T}):")
            print(f"p={psi}")
            print(f"w={w}")
            print(f"u={u}")
            print(f"v={v}")

    def last_sol(self):
        return self.sol_cur
    
    def has_qty_prev(self):
        return self.qty_prev is not None
    def has_qty_prev2(self):
        return self.qty_prev2 is not None
    def last_qty(self):
        return self.qty_cur
    def prev_qty(self):
        return self.qty_prev
    def prev2_qty(self):
        return self.qty_prev2
    
    def calculated_qty(self, i):
        if i == self.cur_step:
            return self.qty_cur
        elif i == self.cur_step-1:
            return self.qty_prev
        elif i == self.cur_step-2:
            return self.qty_prev2
        else:
            # Load from stored csv data
            sol_data = cacheutil.try_load_data(self.sim_step_data_dir(), self.sim_step_data_name(i))
            if sol_data is None:
                raise RuntimeError(f"Not available CSV simulation data for time-step {i}")
            else:
                return self.calc_qties(sol_data)

    def calculated_qty_count(self):
        return self.cur_step + 1
    
    def gen_display_params(self, exclude=[]):
        sim_params = {}
        sim_params["Grid"] = f"{self.Nx} x {self.Ny}"
        sim_params["Dimensions"] = f"{self.Lx} x {self.Ly}"
        sim_params["Time-step (dt)"] = self.dt
        sim_params["Re"] = self.re
        sim_params["Density (rho)"] = self.rho
        sim_params["Viscosity (nu)"] = self.nu

        if "icbc" in self.param:
            sim_params["CI/CC"] = self.param["icbc"]
        elif "icbc-fns" in self.param:
            sim_params["CI/CC"] = "<custom>"

        if "mode" in self.param:
            mode = self.param["mode"]
            sim_params["Mode"] = mode
            if mode.endswith("qubo"):
                if "qubo-R" not in exclude:
                    sim_params["[QUBO] qubits per variable"] = self.param["qubo-R"]
                sim_params["[QUBO] variable encoding"] = self.param["qubo-encoding"]
                sim_params["[QUBO] sample count"] = self.param["qubo-reads"]

                if "qubo-fixed-j0" in self.param:
                    sim_params["[QUBO, fixed] maximum encoding order"] = self.param["qubo-fixed-j0"]

                if "qubo-dyn-psi-base" in self.param:
                    sim_params["[QUBO, dyn] psi minimum increment"] = self.param["qubo-dyn-psi-base"]
                if "qubo-dyn-w-base" in self.param:
                    sim_params["[QUBO, dyn] w minimum increment"] = self.param["qubo-dyn-w-base"]
                if "qubo-dyn-t-order" in self.param:
                    sim_params["[QUBO, dyn] time order"] = self.param["qubo-dyn-t-order"]

        return sim_params

    def plot_steps(self, steps_dir, show_params=False, show_main_title=False):
        if (self.log & LOG_SOLUTIONS) != 0:
            # Show evolution
            """for t in range(self.T+1):
                print(f"Step {t}/{self.T}, solutions:")
                (psi, w, u, v) = self.qty_hist[t]
                print(f"p={psi}")
                print(f"w={w}")
                print(f"u={u}")
                print(f"v={v}")"""

        from cfdplot import plot_steps
        sim_params = self.gen_display_params()
        plot_steps(self, sim_params, show_params, show_main_title, self.Nx, self.Ny, self.Lx, self.Ly, steps_dir)

    def plot_steps_gif(self, steps_dir, gif_file, gif_fps):
        from cfdplot import plot_steps_gif as plot_steps_gif_impl
        plot_steps_gif_impl(self, steps_dir, gif_file, gif_fps)
