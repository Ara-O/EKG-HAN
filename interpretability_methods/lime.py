import sys

sys.path.append("../")
import numpy as np
import torch
from captum.attr import Lime
import torch.nn.functional as F
from utils.set_seed import set_seed
import matplotlib.pyplot as plt

 
def run_lime(model, x, y_true_description, y_true, rr=None, baseline='zero',
             rr_mode='real',          # 'real' | 'neutral_zero' | 'neutral_mean' | 'joint'
             full=None, r_peaks=None, pre_samples=99, post_samples=201, leads=None):
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

    feature_mask = torch.zeros_like(x, dtype=torch.long)
    feature_count = 0

    if is_4d:
        B, n_beats, n_time, n_channels = x.shape
        chunks_per_beat = 6
        chunk_len = n_time // chunks_per_beat
        for b in range(n_beats):
            for ch in range(n_channels):
                for i in range(chunks_per_beat):
                    t_start = i * chunk_len
                    t_end = min((i + 1) * chunk_len, n_time)
                    feature_mask[:, b, t_start:t_end, ch] = feature_count
                    feature_count += 1
    else:
        B, dim1, dim2 = x.shape
        if dim1 == 12:
            n_channels, n_time = dim1, dim2
            chunk_len = 50
            for ch in range(n_channels):
                for i in range(n_time // chunk_len):
                    t_start = i * chunk_len
                    t_end = min((i + 1) * chunk_len, n_time)
                    feature_mask[:, ch, t_start:t_end] = feature_count
                    feature_count += 1
        else:
            n_time, n_channels = dim1, dim2
            chunk_len = 50
            for ch in range(n_channels):
                for i in range(n_time // chunk_len):
                    t_start = i * chunk_len
                    t_end = min((i + 1) * chunk_len, n_time)
                    feature_mask[:, t_start:t_end, ch] = feature_count
                    feature_count += 1

    ecg_feature_count = feature_count
    print(f"ECG LIME features: {ecg_feature_count}")

    uses_rr = rr is not None

    if uses_rr:
        def model_forward_joint(input_tensor, rr_tensor):
            return model(input_tensor, rr_tensor)
    else:
        def model_forward_single(input_tensor):
            return model(input_tensor)

    # Prediction always uses real rr
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

    print("Computing LIME attributions...")
    rr_attributions = None  # populated only in 'joint' mode

    if not uses_rr:
        lime = Lime(model_forward_single)
        attributions = lime.attribute(
            inputs=x, target=y_pred,
            feature_mask=feature_mask, baselines=baseline_x,
            n_samples=50000, perturbations_per_eval=500, show_progress=True,
        )

    elif rr_mode == 'real':
        # RR held at real value during perturbations (side input, not perturbed)
        lime = Lime(model_forward_joint)
        attributions = lime.attribute(
            inputs=x, target=y_pred,
            feature_mask=feature_mask, baselines=baseline_x,
            additional_forward_args=(rr,),
            n_samples=50000, perturbations_per_eval=500, show_progress=True,
        )

    elif rr_mode in ('neutral_zero', 'neutral_mean'):
        # RR held at a neutral value during perturbations
        if rr_mode == 'neutral_zero':
            rr_neutral = torch.zeros_like(rr)
        else:
            rr_neutral = torch.full_like(rr, rr.mean().item())

        lime = Lime(model_forward_joint)
        attributions = lime.attribute(
            inputs=x, target=y_pred,
            feature_mask=feature_mask, baselines=baseline_x,
            additional_forward_args=(rr_neutral,),
            n_samples=50000, perturbations_per_eval=500, show_progress=True,
        )

    elif rr_mode == 'joint':
        # Perturb both ECG chunks and RR beats; each RR beat is one feature
        rr_mask = torch.zeros_like(rr, dtype=torch.long)
        n_beats_rr = rr.shape[1]
        for b in range(n_beats_rr):
            rr_mask.select(1, b).fill_(ecg_feature_count + b)
        print(f"RR LIME features (one per beat): {n_beats_rr}")
        print(f"Total LIME features: {ecg_feature_count + n_beats_rr}")

        baseline_rr = torch.full_like(rr, rr.mean().item())

        lime = Lime(model_forward_joint)
        attrs = lime.attribute(
            inputs=(x, rr), target=y_pred,
            feature_mask=(feature_mask, rr_mask),
            baselines=(baseline_x, baseline_rr),
            n_samples=50000, perturbations_per_eval=500, show_progress=True,
        )
        attributions, rr_attributions = attrs

    else:
        raise ValueError(f"Unknown rr_mode: {rr_mode}")

    ecg_leads, attr_leads = [], []

    for lead_idx in range(n_leads):
        if is_4d and full is not None and r_peaks is not None:
            full_len = full.shape[2]
            attr_full = np.zeros(full_len)
            ecg_sig   = full[0, lead_idx, :].cpu().detach().numpy()

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

            attr_n = attr_full / np.max(np.abs(attr_full)) if np.max(np.abs(attr_full)) > 0 else attr_full
        else:
            if x.shape[1] == 12:
                ecg_sig = x[0, lead_idx, :].cpu().detach().numpy()
                attr_n  = attributions[0, lead_idx, :].cpu().detach().numpy()
            else:
                ecg_sig = x[0, :, lead_idx].cpu().detach().numpy()
                attr_n  = attributions[0, :, lead_idx].cpu().detach().numpy()

            attr_n = attr_n / np.max(np.abs(attr_n)) if np.max(np.abs(attr_n)) > 0 else attr_n

        ecg_leads.append(ecg_sig)
        attr_leads.append(attr_n)

 
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
        ax       = axes[plot_idx]
        ecg_sig  = ecg_leads[lead_idx]
        attr_n   = attr_leads[lead_idx]
        time_steps = np.arange(len(ecg_sig))

        ecg_min, ecg_max = np.min(ecg_sig), np.max(ecg_sig)
        if ecg_min == ecg_max:
            ecg_min -= 0.1
            ecg_max += 0.1

        margin = (ecg_max - ecg_min) * 0.05
        ax.set_ylim(ecg_min - margin, ecg_max + margin)

        ax.plot(time_steps, ecg_sig, 'k-', alpha=0.6, linewidth=0.8)

        extent = [time_steps[0], time_steps[-1], ecg_min - margin, ecg_max + margin]
        im = ax.imshow(
            attr_n[np.newaxis, :],
            aspect='auto', extent=extent,
            cmap='seismic', alpha=0.45, vmin=-1, vmax=1
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

    axes[-1].set_xlabel('Time Step', fontsize=20, fontfamily='serif')

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

    plt.show()
    fig.savefig('lime_plot.png', dpi=300, bbox_inches='tight')

    for lead_idx in leads_to_plot:
        attr_n = attr_leads[lead_idx]
        print(f"Lead {lead_names[lead_idx]:>3s} | "
              f"Range [{attr_n.min():.3f}, {attr_n.max():.3f}] | "
              f"Mean |attr|: {np.mean(np.abs(attr_n)):.4f}")

    print(f"\nRR mode: {rr_mode if uses_rr else 'n/a'}")
    print(f"Actual label {y_true} - {y_true_description}. Predicted {y_pred}")

    # Joint only: RR vs ECG contribution split
    if rr_attributions is not None:
        ecg_mag = attributions.abs().sum().item()
        rr_mag  = rr_attributions.abs().sum().item()
        total   = ecg_mag + rr_mag if (ecg_mag + rr_mag) > 0 else 1.0
        print(f"\n--- Joint LIME attribution split ---")
        print(f"ECG contributes {100 * ecg_mag / total:.1f}% of total |attribution|")
        print(f"RR  contributes {100 * rr_mag  / total:.1f}% of total |attribution|")
        rr_per_beat = rr_attributions.squeeze().cpu().detach().numpy()
        print(f"RR attribution per beat: {np.array2string(rr_per_beat, precision=4)}")

        return attr_leads, rr_per_beat

    return attr_leads