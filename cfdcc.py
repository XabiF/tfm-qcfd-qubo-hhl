import cfd
import numpy as np

#############################################################################
# Lid-driven cavity

ulid = 10.0

def lid_uv_bc(sim: cfd.Simulation, x, y):
    if (x == 1) or (y == 1) or (x == sim.Nx):
        return (0.0, 0.0)
    elif y == sim.Ny:
        return (ulid, 0.0)
    else:
        return None

def lid_uv_ic(sim: cfd.Simulation, x, y):
    return (0.0, 0.0)

def lid_psi0(sim: cfd.Simulation, x, y):
    # Avoid steep gradients in initial values later, and convergence to unwanted stationary states
    return 0.0

def lid_char(sim: cfd.Simulation):
    return (ulid, np.sqrt(sim.Ly*sim.Lx))

###########################################################################
# Obstacle in horizontal flow

ubase = 0.1

def obst_uv_bc(sim: cfd.Simulation, x, y):
    if (x == 1) or (y == 1) or (x == sim.Nx) or (y == sim.Ny):
        return (ubase, 0.0)
    
    obst_fn = sim.param["bc-obst-obst"]
    if obst_fn(x, y):
        return (0.0, 0.0)
    
    return None

def obst_uv_ic(sim: cfd.Simulation, x, y):
    obst_fn = sim.param["bc-obst-obst"]
    if obst_fn(x, y):
        return (0.0, 0.0)
    
    return (ubase, 0.0)

def obst_psi0(sim: cfd.Simulation, x, y):
    xp = (x-1)*sim.dx
    yp = (y-1)*sim.dy
    (u, v) = obst_uv_ic(sim, x, y)
    return u*yp - v*xp

def obst_char(sim: cfd.Simulation):
    obst_fn = sim.param["bc-obst-obst"]
    num_obst_points = 0
    for i in range(sim.Nx):
        for j in range(sim.Ny):
            x = i+1
            y = j+1
            if obst_fn(x, y):
                num_obst_points += 1
    obst_size = np.sqrt(obst_fn)        
    return (ubase, obst_size)

#######################################################################
# Taylor-Green vortices

def tgv_uv_bc(sim: cfd.Simulation, x, y):
    # No edges fixed by BC here
    return None

TGV_U = 3.0
TGV_RATIO = 2

def tgv_u0_v0(sim: cfd.Simulation):
    u0 = TGV_U
    v0 = TGV_U*(sim.Lx/sim.Ly)
    return (u0, v0)

def tgv_psicoef(sim: cfd.Simulation):
    return TGV_U/(TGV_RATIO*np.pi/sim.Lx)

def tgv_uv_ic(sim: cfd.Simulation, x, y):
    global TGV_RATIO

    xp = (x-1)*sim.dx
    yp = (y-1)*sim.dy

    u0, v0 = tgv_u0_v0(sim)

    u = u0*np.sin(TGV_RATIO*xp*np.pi/sim.Lx)*np.cos(TGV_RATIO*yp*np.pi/sim.Ly)
    v = - v0*np.cos(TGV_RATIO*xp*np.pi/sim.Lx)*np.sin(TGV_RATIO*yp*np.pi/sim.Ly)
    return (u, v)

def tgv_psi0(sim: cfd.Simulation, x, y):
    global TGV_RATIO

    xp = (x-1)*sim.dx
    yp = (y-1)*sim.dy
    return tgv_psicoef(sim)*np.sin(TGV_RATIO*xp*np.pi/sim.Lx)*np.sin(TGV_RATIO*yp*np.pi/sim.Ly)

def tgv_char(sim: cfd.Simulation):
    L = np.sqrt(sim.Lx*sim.Ly)
    l = L/(np.pi*TGV_RATIO)
    u0, v0 = tgv_u0_v0(sim)
    u = np.sqrt(u0*v0)
    return (u, l)

##############################
##############################

def search_ic_bc(name):
    if name == "ldc":
        return (lid_uv_ic, lid_uv_bc, lid_psi0, lid_char)
    elif name == "obst":
        return (obst_uv_ic, obst_uv_bc, obst_psi0, obst_char)
    elif name == "tgv":
        return (tgv_uv_ic, tgv_uv_bc, tgv_psi0, tgv_char)
    
    ################################
    else:
        raise RuntimeError(f"Invalid requested built-in IC/BC '{name}'!")
