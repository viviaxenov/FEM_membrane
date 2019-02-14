import numpy as np
import matplotlib.pyplot as plt

sample_u = np.load('../res/conv/sample_u.npy')
sample_v = np.load('../res/conv/sample_v.npy')

d_u = sample_u[1:] - sample_u[:-1]
d_v = sample_v[1:] - sample_v[:-1]

diffs = np.hstack((d_u, d_v))
norms = np.linalg.norm(diffs, axis=1)

xs = np.arange(1,6)

fig, axs = plt.subplots(1,2)

ax = axs[0]
ax.plot(xs, norms, 'bs')
ax.grid(True)

ax = axs[1]
ax.plot(xs, np.log2(norms), 'rs-')
ax.grid(True)

fig.savefig('../res/conv.png', fmt='png')

print(norms)

