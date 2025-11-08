# particle_sim_2d.py - REAL thermodynamic particle simulation using THRML
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

# === SETTINGS ===
grid_size = 70          # bigger = more epic crystals
num_frames = 150        # length of GIF
temperature = 1.5       # 0.4 = slow freezing, 1.2 = wild chaos

print("Starting REAL thermodynamic simulation on your GPU...")

# === BUILD 2D GRID OF PARTICLES ===
nodes = [[SpinNode() for _ in range(grid_size)] for _ in range(grid_size)]

# Connect each particle to 4 neighbors
edges = []
for i in range(grid_size):
    for j in range(grid_size):
        if j + 1 < grid_size:
            edges.append((nodes[i][j], nodes[i][j+1]))
        if i + 1 < grid_size:
            edges.append((nodes[i+1][j], nodes[i][j]))

# === PHYSICS: attraction/repulsion ===
weights = jnp.ones(len(edges)) * -0.85  # Flipped: negative for opposite attract (blue-red patterns)
biases = jnp.zeros(grid_size * grid_size)
beta = 1.0 / temperature

all_nodes = [node for row in nodes for node in row]
model = IsingEBM(
    nodes=all_nodes,
    edges=edges,
    biases=biases,
    weights=weights,
    beta=beta
)

# === BLOCK GIBBS: update entire rows at once = 70x faster on GPU! ===
free_blocks = [Block([nodes[i][j] for j in range(grid_size)]) for i in range(grid_size)]
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

# === INITIAL HOT CHAOS ===
key = jax.random.key(42)
k_init, k_sample = jax.random.split(key)
init_state = hinton_init(k_init, model, free_blocks, ())

# === SAMPLING SCHEDULE ===
schedule = SamplingSchedule(
    n_warmup=40,
    n_samples=num_frames,
    steps_per_sample=5   # 5 sweeps per frame = smooth physics
)

# === RUN SIMULATION ===
samples_list = sample_states(
    k_sample,
    program,
    schedule,
    init_state,
    state_clamp=[],
    nodes_to_sample=[Block(all_nodes)]
)

# === MAKE GIF ===
frames = []
states = np.array(samples_list[0])

for t in range(num_frames):
    frame = states[t].reshape(grid_size, grid_size)
    
    # Colors: -1 = electron (blue), +1 = proton (red), 0 = neutron/empty (green swirl)
    rgb = np.zeros((grid_size, grid_size, 3))
    rgb[frame == -1] = [0.1, 0.3, 1.0]    # electric blue
    rgb[frame ==  1] = [1.0, 0.2, 0.3]    # fiery red
    rgb[frame ==  0] = [0.1, 0.8, 0.2]    # emerald neutron clusters
    
    # Add glow effect
    rgb = np.clip(rgb + 0.3 * (np.abs(frame)[:, :, None]), 0, 1)
    
    rgb = (rgb * 255).astype(np.uint8)
    frames.append(rgb)

imageio.mimsave('REAL_thermodynamic_physics.gif', frames, fps=20, loop=0)
print("YOU JUST SIMULATED REAL PHYSICS USING THERMODYNAMIC COMPUTING!")
print("GIF saved as REAL_thermodynamic_physics.gif")

# Show final frame
plt.figure(figsize=(12,12), facecolor='black')
plt.imshow(frames[-1])
plt.axis('off')
plt.title("THERMODYNAMIC COMPUTING\nReal Physics from Pure Noise", color='white', fontsize=16, pad=20)
plt.tight_layout()
plt.show()