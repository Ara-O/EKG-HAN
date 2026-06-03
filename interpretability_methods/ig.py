import sys

sys.path.append("../")
import numpy as np
import torch
from captum.attr import IntegratedGradients
import torch.nn.functional as F
from utils.set_seed import set_seed
import matplotlib.pyplot as plt

def aggregate_attributions(attr, window_size):
    n_windows = len(attr) // window_size
    # Truncate to fit complete windows
    attr_truncated = attr[:n_windows * window_size]
    # Reshape and sum over each window
    attr_reshaped = attr_truncated.reshape(n_windows, window_size)
    return np.sum(attr_reshaped, axis=1)


def aggregate_signal(signal, window_size):
    n_windows = len(signal) // window_size
    # Truncate to fit complete windows
    signal_truncated = signal[:n_windows * window_size]
    # Reshape and average over each window
    signal_reshaped = signal_truncated.reshape(n_windows, window_size)
    return np.mean(signal_reshaped, axis=1)

mapping = {
    0: 'Overall interpretation',
    1: 'Sinus node rhythms',
    2: 'Supraventricular arrhythmias',
    3: 'Supraventricular tachyarrhythmias',
    4: 'Ventricular arrhythmias',
    5: 'Atrioventricular conduction',
    6: 'Intraventricular and intra-atrial conduction',
    7: 'Axis and voltage',
    8: 'Chamber hypertrophy or enlargement',
    9: 'ST segment, T wave, and U wave',
    10: 'Myocardial infarction',
}

def run_ig(model, x, y_true_description, y_true, rr=None, baseline='zero',
           rr_mode='real',          # 'real' | 'neutral_zero' | 'neutral_mean' | 'joint'
           full=None, r_peaks=None, pre_samples=99, post_samples=201,
           window_size=50, leads=None):
    import matplotlib
    matplotlib.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

    is_4d = len(x.shape) == 4
    if is_4d:
        model.train()
    else:
        model.eval()

    n_leads = 12
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    if leads is None:
        leads_to_plot = list(range(n_leads))
    else:
        leads_to_plot = []
        for l in leads:
            if isinstance(l, str):
                if l not in lead_names:
                    raise ValueError(f"Unknown lead name: '{l}'. Valid names: {lead_names}")
                leads_to_plot.append(lead_names.index(l))
            elif isinstance(l, int):
                if not (0 <= l < n_leads):
                    raise ValueError(f"Lead index {l} out of range (0-{n_leads - 1})")
                leads_to_plot.append(l)
            else:
                raise TypeError(f"Lead must be str or int, got {type(l)}")

    uses_rr = rr is not None

    if uses_rr:
        def model_forward_joint(input_tensor, rr_tensor):
            return model(input_tensor, rr_tensor)
    else:
        def model_forward_single(input_tensor):
            return model(input_tensor)

    with torch.no_grad():
        if uses_rr:
            y_pred = torch.argmax(model_forward_joint(x, rr), dim=1).item()
        else:
            y_pred = torch.argmax(model_forward_single(x), dim=1).item()

    # ECG baseline
    if baseline == 'zero':
        baseline_x = torch.zeros_like(x)
    elif baseline == 'noise':
        baseline_x = torch.randn_like(x) * 0.01
    else:
        raise ValueError(f"Unknown baseline: {baseline}")

    print("Computing Integrated Gradients attributions...")
    rr_attributions = None  # populated only in 'joint' mode

    if not uses_rr:
        ig = IntegratedGradients(model_forward_single)
        attributions, _ = ig.attribute(
            inputs=x, baselines=baseline_x, target=y_pred,
            return_convergence_delta=True,
        )

    elif rr_mode == 'real':
        ig = IntegratedGradients(model_forward_joint)
        attributions, _ = ig.attribute(
            inputs=x, baselines=baseline_x, target=y_pred,
            additional_forward_args=(rr,),
            return_convergence_delta=True,
        )

    elif rr_mode in ('neutral_zero', 'neutral_mean'):
        if rr_mode == 'neutral_zero':
            rr_neutral = torch.zeros_like(rr)
        else:  # neutral_mean
            rr_neutral = torch.full_like(rr, rr.mean().item())

        ig = IntegratedGradients(model_forward_joint)
        attributions, _ = ig.attribute(
            inputs=x, baselines=baseline_x, target=y_pred,
            additional_forward_args=(rr_neutral,),
            return_convergence_delta=True,
        )

    elif rr_mode == 'joint':
        baseline_rr = torch.full_like(rr, rr.mean().item())
        ig = IntegratedGradients(model_forward_joint)
        attrs, _ = ig.attribute(
            inputs=(x, rr), baselines=(baseline_x, baseline_rr),
            target=y_pred, return_convergence_delta=True,
        )
        attributions, rr_attributions = attrs  

    else:
        raise ValueError(f"Unknown rr_mode: {rr_mode}")

    all_ecg, all_attr_norm, all_attr_aggregated, all_attr_expanded = [], [], [], []

    for lead_idx in range(n_leads):
        if is_4d and full is not None and r_peaks is not None:
            full_len = full.shape[2]
            attr_full = np.zeros(full_len)
            ecg_full  = full[0, lead_idx, :].cpu().detach().numpy()

            for i, peak in enumerate(r_peaks):
                start = max(0, peak - pre_samples)
                end   = min(full_len, peak + post_samples)
                beat_attr = attributions[0, i, :, lead_idx].cpu().detach().numpy()
                beat_attr_interp = np.interp(
                    np.arange(end - start),
                    np.linspace(0, end - start, len(beat_attr)),
                    beat_attr
                )
                attr_full[start:end] = beat_attr_interp
        else:
            if x.shape[1] == 12:
                ecg_full  = x[0, lead_idx, :].cpu().detach().numpy()
                attr_full = attributions[0, lead_idx, :].cpu().detach().numpy()
            else:
                ecg_full  = x[0, :, lead_idx].cpu().detach().numpy()
                attr_full = attributions[0, :, lead_idx].cpu().detach().numpy()

        attr_aggregated = aggregate_attributions(attr_full, window_size)
        attr_expanded   = np.repeat(attr_aggregated, window_size)[:len(ecg_full)]
        attr_norm = (attr_expanded / np.max(np.abs(attr_expanded))
                     if np.max(np.abs(attr_expanded)) > 0 else attr_expanded)

        all_ecg.append(ecg_full)
        all_attr_norm.append(attr_norm)
        all_attr_aggregated.append(attr_aggregated)
        all_attr_expanded.append(attr_expanded)

    n_plot = len(leads_to_plot)
    fig, axes = plt.subplots(
        n_plot, 1,
        figsize=(24, 2.8 * n_plot),
        sharex=True,
        gridspec_kw={'hspace': 0.08}
    )
    if n_plot == 1:
        axes = [axes]

    for plot_idx, lead_idx in enumerate(leads_to_plot):
        ax        = axes[plot_idx]
        ecg_full  = all_ecg[lead_idx]
        attr_norm = all_attr_norm[lead_idx]
        time_steps = np.arange(len(ecg_full))

        ecg_min, ecg_max = np.min(ecg_full), np.max(ecg_full)
        if ecg_min == ecg_max:
            ecg_min -= 0.1
            ecg_max += 0.1

        margin = (ecg_max - ecg_min) * 0.05
        ax.set_ylim(ecg_min - margin, ecg_max + margin)

        ax.plot(time_steps, ecg_full, 'k-', alpha=0.6, linewidth=0.8)

        im = ax.pcolormesh(
            [time_steps, time_steps],
            [ecg_min - margin, ecg_max + margin],
            [attr_norm, attr_norm],
            cmap='seismic', alpha=0.4, vmin=-1, vmax=1
        )

        ax.text(
            0.002, 0.95, lead_names[lead_idx],
            transform=ax.transAxes,
            fontsize=28, fontweight='bold', va='top', ha='left',
            fontfamily='serif',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6, ec='none')
        )

        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=20)

        if plot_idx == n_plot - 1:
            fig.canvas.draw()
            bbox_top = axes[0].get_position()
            bbox_bot = axes[-1].get_position()
            cbar_width, cbar_pad = 0.015, 0.01
            cbar_left   = bbox_top.x1 + cbar_pad
            cbar_bottom = bbox_bot.y0
            cbar_height = bbox_top.y1 - bbox_bot.y0
            cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Importance', fontsize=20, fontfamily='serif')
            cbar.ax.tick_params(labelsize=20)

    axes[-1].set_xlabel('Time Step', fontsize=20, fontfamily='serif')

    plt.show()
    fig.savefig('ig_plot.png', dpi=300, bbox_inches='tight')

    attr_norm_ii = all_attr_norm[1]
    print(f"Attribution Range (Lead II): {attr_norm_ii.min():.4f} to {attr_norm_ii.max():.4f}")
    print(f"Mean Abs Attribution (Lead II): {np.mean(np.abs(attr_norm_ii)):.4f}")
    print(f"Number of windows: {len(all_attr_aggregated[1])}")
    print(f"Window size: {window_size} samples")
    print(f"RR mode: {rr_mode if uses_rr else 'n/a'}")
    print(f"Actual label {y_true} - {y_true_description}. Predicted {y_pred}")

    if rr_attributions is not None:
        ecg_mag = attributions.abs().sum().item()
        rr_mag  = rr_attributions.abs().sum().item()
        total   = ecg_mag + rr_mag if (ecg_mag + rr_mag) > 0 else 1.0
        print(f"\n--- Joint attribution split ---")
        print(f"ECG contributes {100 * ecg_mag / total:.1f}% of total |attribution|")
        print(f"RR  contributes {100 * rr_mag  / total:.1f}% of total |attribution|")
        rr_per_beat = rr_attributions.squeeze().cpu().detach().numpy()
        print(f"RR attribution per beat: {np.array2string(rr_per_beat, precision=4)}")

    if rr_attributions is not None:
        return all_attr_expanded, rr_attributions.squeeze().cpu().detach().numpy()
    return all_attr_expanded