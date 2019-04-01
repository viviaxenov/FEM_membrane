import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'stix'


fig, axs = plt.subplots(3,1, sharex=True)
fig.suptitle('Распределенная нагрузка $\propto \cos^2(r)$')

for idx, ord in enumerate([1, 2, np.inf]):
    splits = []
    diffs = []
    for k in range(6):
        sample_prev = np.append(np.load('./sample_u{0:d}.npy'.format(k)), np.load('./sample_v{0:d}.npy'.format(k)))
        sample_next = np.append(np.load('./sample_u{0:d}.npy'.format(k + 1)), np.load('./sample_v{0:d}.npy'.format(k + 1)))
        diff = np.linalg.norm(sample_next - sample_prev, ord=ord)
        splits += [k]
        diffs += [diff]

    splits = np.array(splits)
    diffs = np.array(diffs)

    p = np.polyfit(splits[1:], np.log2(diffs)[1:], deg=1)
    line = np.poly1d(p)

    xs = np.linspace(0, 5, endpoint=True)
    curve = np.exp(p[1] + p[0]*xs)


#    ax = axs[idx, 0]
#
#    ax.set_xlim(left=-0.1, right=5.1)
#
#    ax.plot(splits, diffs, 'rs')
#    #ax.plot(xs, curve, 'b-')
#    ax.grid(True)
#    ax.set_xlabel('$k$ - number of grid splits')
#    ax.set_ylabel('$||x_{{k + 1}} - x_k||_{0}$'.format(ord))

    #ax.annotate('Omitted from fit', xy=(splits[0], diffs[0]), xytext=(1, 1), arrowprops=dict(arrowstyle='->'))

    ax = axs[idx]
    ax.plot(splits, np.log2(diffs), 'rs')
    latex_text = r'$\frac{d(\log_2{||x_{k + 1} - x_k ||})}{dk} = '
    formated_text = '{0:.3f}$'.format(p[0])

    ax.plot(splits, line(splits), 'b-', label=(latex_text + formated_text))
    ax.set_xlabel('$k$ - число разбиений сетки')
    ax.set_ylabel('$\log_2{{||x_{{k + 1}} - x_k||_{{ {0} }} }}$'.format(ord))

    ax.legend()
    ax.grid(True)
#    ax.annotate('Omitted from fit', xy=(splits[0], np.log2(diffs[0])), xytext=(splits[0] + 1, np.log(diffs[0]) + 3), arrowprops=dict(arrowstyle='->'))

fig.tight_layout()
fig.savefig('./plots.png', fmt='png')

plt.show()


