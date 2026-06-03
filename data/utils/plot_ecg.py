import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_ecg_leads(x):
    matplotlib.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'path.simplify': False,        # preserve every ECG sample
        'agg.path.chunksize': 10000,   # avoid glitches on long traces
        'lines.antialiased': True,
        'text.antialiased': True,
        'font.weight': 'normal',
        'axes.titleweight': 'normal',
        'axes.labelweight': 'normal',
        'figure.titleweight': 'normal',
    })

    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
                  'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    n_leads = 12

    # Figure height scales with number of leads; matches the IG/LIME stacks
    fig, axes = plt.subplots(
        n_leads, 1,
        figsize=(20, 2.0 * n_leads),
        sharex=True,
        gridspec_kw={'hspace': 0.08},
    )

    for i, ax in enumerate(axes):
        if x.shape[1] == 12:   # channels-first  (B, 12, T)
            signal = x[0, i, :].cpu().detach().numpy()
        else:                  # channels-last   (B, T, 12)
            signal = x[0, :, i].cpu().detach().numpy()

        time_steps = np.arange(signal.shape[0])

        s_min, s_max = float(signal.min()), float(signal.max())
        if s_min == s_max:
            s_min -= 0.1
            s_max += 0.1
        margin = (s_max - s_min) * 0.05
        ax.set_ylim(s_min - margin, s_max + margin)

        ax.plot(time_steps, signal, 'k-', linewidth=0.8)

        ax.text(
            0.002, 0.95, lead_names[i],
            transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left',
            fontfamily='serif',
            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                      alpha=0.6, ec='none'),
        )

        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=13)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xlabel('Time Step', fontsize=15, fontfamily='serif')
    plt.suptitle('ECG — All 12 Leads', fontsize=16, fontfamily='serif', y=0.905)
    plt.show()