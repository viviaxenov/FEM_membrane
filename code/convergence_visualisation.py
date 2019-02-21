import numpy as np
import matplotlib.pyplot as plt


splits = []
diffs = []

for k in range(6):
    sample_prev = np.append(np.load(f'sample_u{k:d}.npy'), np.load(f'sample_v{k:d}'))
    sample_next = np.append(np.load(f'sample_u{k + 1:d}.npy'), np.load(f'sample_v{k + 1:d}'))
    diff = np.linalg.norm(sample_next - sample_prev, ord=2)
    splits += [k]
    diffs += [diff]

splits = np.array(splits)
diffs = np.array(diffs)

fig, axs = plt.subplots(1,2)

ax = axs[0]
ax.plot(splits, diffs, 'bs')
ax.grid(True)

ax = axs[1]
ax.plot(splits, np.log2(diffs), 'rs-')
ax.grid(True)

fig.savefig('../res/conv.png', fmt='png')

print(diffs)

