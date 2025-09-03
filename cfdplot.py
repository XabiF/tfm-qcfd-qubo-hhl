import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display, HTML
import numpy as np
import os
import imageio
import dwave_networkx as dnx
import networkx as nx

NORMALIZATION_NONE = 1
NORMALIZATION_0_1 = 2
NORMALIZATION_M1_1 = 3

def plot_scalar(fig, ax, s_vec, norm, cmap, Nx, Ny, Lx, Ly, title, legend):
    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)
    
    if norm == NORMALIZATION_M1_1:
        max_abs_s = max([abs(s) for s in s_vec])
        if max_abs_s == 0:
            s_norm_vec = s_vec
        s_norm_vec = np.array([s/max_abs_s for s in s_vec])

        s_2d = s_norm_vec.reshape((Nx, Ny)).T
        c = ax.pcolormesh(x, y, s_2d, cmap=cmap, shading='auto', vmin=-1, vmax=1)
    elif norm == NORMALIZATION_0_1:
        max_s = max(s_vec)
        min_s = min(s_vec)
        
        s_2d = s_vec.reshape((Nx, Ny)).T
        if max_s != min_s:
            s_map_2d = (s_2d - min_s) / (max_s - min_s)
        else:
            s_map_2d = s_2d

        c = ax.pcolormesh(x, y, s_map_2d, cmap=cmap, shading='auto', vmin=0, vmax=1)
    else:
        s_2d = s_vec.reshape((Nx, Ny)).T
        c = ax.pcolormesh(x, y, s_2d, cmap=cmap, shading='auto', vmin=min(s_vec), vmax=max(s_vec))

    # Add colorbar
    fig.colorbar(c, ax=ax, label=legend)

    # Slightly above 1 for the plot to mark limits nicely
    ax.set_xticks(np.arange(0, 1.1, 1))
    ax.set_yticks(np.arange(0, 1.1, 1))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    # Keep axes visible but remove box border
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Keep tick marks and labels
    ax.tick_params(direction='out', length=5)

def plot_vector(fig, ax, x_vec, y_vec, cmap, Nx, Ny, Lx, Ly, title, legend):
    assert len(x_vec) == len(y_vec)
    mag_vec = np.array([np.sqrt(x_vec[i]**2 + y_vec[i]**2) for i in range(len(x_vec))])
    max_mag = np.max(mag_vec)
    if max_mag == 0:
        mag_norm_vec = mag_vec
    mag_norm_vec = np.array([m/max_mag for m in mag_vec])
    mag_norm_2d = mag_vec.reshape((Nx, Ny)).T

    x_2d = x_vec.reshape((Nx, Ny)).T
    y_2d = y_vec.reshape((Nx, Ny)).T

    x = np.linspace(0, Lx, Nx + 1)
    y = np.linspace(0, Ly, Ny + 1)

    c = ax.pcolormesh(x, y, mag_norm_2d, cmap=cmap, shading='auto', vmin=0, vmax=max_mag)

    vec_factor_x = 0.75 / Nx
    vec_factor_y = 0.75 / Ny
    for i in range(Nx):
        for j in range(Ny):
            x_val = x_2d[j,i]
            y_val = y_2d[j,i]
            xy_norm = np.sqrt(x_val**2 + y_val**2)
            if xy_norm != 0:
                x_val /= xy_norm
                y_val /= xy_norm
                x_val *= vec_factor_x
                y_val *= vec_factor_y

                len_x = x[i+1] - x[i]
                len_y = y[j+1] - y[j]
                arrow_x = x[i] + (len_x - x_val) / 2
                arrow_y = y[j] + (len_y - y_val) / 2
                ax.quiver(arrow_x, arrow_y, x_val, y_val, angles='xy', scale_units='xy', scale=1, color="white")

    # Add colorbar
    fig.colorbar(c, ax=ax, label=legend)

    # Slightly above 1 for the plot to mark limits nicely
    ax.set_xticks(np.arange(0, 1.1, 1))
    ax.set_yticks(np.arange(0, 1.1, 1))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    # Keep axes visible but remove box border
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Keep tick marks and labels
    ax.tick_params(direction='out', length=5)

def plot_step(cant, Nx, Ny, Lx, Ly, title, params, show_params, show_main_title, step_file):
    (psi_vec, w_vec, u_vec, v_vec) = cant

    ax_count = 4 if show_params else 3
    figwidth = 16 if show_params else 12

    fig, axes = plt.subplots(1, ax_count, figsize=(figwidth, 4))

    plot_vector(fig, axes[0], u_vec, v_vec, "plasma", Nx, Ny, Lx, Ly, "Flow velocity field", "Velocity magnitude")
    plot_scalar(fig, axes[1], w_vec, NORMALIZATION_NONE, "viridis", Nx, Ny, Lx, Ly, "Flow vorticity", "Vorticity")
    plot_scalar(fig, axes[2], psi_vec, NORMALIZATION_NONE, "magma", Nx, Ny, Lx, Ly, "Flow streamlines", "Streamline values")

    if show_params:
        axes[3].axis("off")
        cell_text = [[k, v] for k, v in params.items()]
        table = axes[3].table(cellText=cell_text, colLabels=["Property", "Value"], loc='center')
        table.set_fontsize(plt.rcParams['font.size'])
        table.scale(1.5, 1.5)
        for cell in table.get_celld().values():
            cell.get_text().set_fontsize(plt.rcParams['font.size'])

    if show_main_title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(step_file)
    plt.close()

def format_step_frame_i(i):
    return f"step-{i:03d}.png"

def plot_steps(sim, params, show_params, show_main_title, Nx, Ny, Lx, Ly, steps_dir):
    if not os.path.exists(steps_dir):
        os.makedirs(steps_dir)

    qty_count = sim.calculated_qty_count()
    for t in range(qty_count):
        step_file = os.path.join(steps_dir, format_step_frame_i(t))
        step_title = f"Flow simulation [{t}/{qty_count}]"
        cant = sim.calculated_qty(t)
        plot_step(cant, Nx, Ny, Lx, Ly, step_title, params, show_params, show_main_title, step_file)

def plot_steps_gif(sim, steps_dir, gif_file, gif_fps):
    step_images = [imageio.imread(os.path.join(steps_dir, format_step_frame_i(t))) for t in range(sim.calculated_qty_count())]
    imageio.mimsave(gif_file, step_images, fps=gif_fps)

def display_image(path):
    img = mpimg.imread(path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def display_step_frame(steps_dir, t):
    img = mpimg.imread(os.path.join(steps_dir, format_step_frame_i(t)))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def display_gif(path):
    display(HTML(f'<img src="{path}">'))

def plot_qubo_chimera(N, R, bit_tensor):
    # Figura tipo Chimera A: representar activación de bits
    chimera_graph = dnx.chimera_graph(5, 4, 4, coordinates=True)  # suficiente para Ny * R qubits
    pos = dnx.chimera_layout(chimera_graph, dim=2, scale=1, center=(0, 0))

    # Lista de nodos activos
    active_nodes = []
    flat_index = 0
    for i in range(N):
        for j in range(R):
            for k in range(2):
                if bit_tensor[i, j, k] == 1:
                    if flat_index < len(chimera_graph.nodes):
                        active_nodes.append(list(chimera_graph.nodes)[flat_index])
                flat_index += 1

    # Dibujar el grafo y destacar bits activos
    def node_color(n):
        return 'lime' if n in active_nodes else 'lightgray'

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(chimera_graph, pos, edge_color='lightgray', alpha=0.5)
    nx.draw_networkx_nodes(chimera_graph, pos, node_color=[node_color(n) for n in chimera_graph.nodes], node_size=120)
    plt.title("Figura tipo Chimera Graph del hardware de D-Wave (activación QUBO)")
    plt.axis('off')
    plt.show()

def plot_sparsity(A):
    plt.figure(figsize=(8,8))
    plt.spy(A, markersize=10)
    plt.title('Sparsity of matrix A')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.grid(False)
    plt.show()
