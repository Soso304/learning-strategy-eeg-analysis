"""
EEG Processing Pipeline
========================
Full pipeline: re-reference -> filter -> ocular correction -> segmentation ->
baseline correction -> artifact rejection -> CWT (Morlet) -> spectral amplitude/power

Analysis stage: topomaps and time-frequency tables split by CONDITION and STRESS GROUP.

Requirements:
    pip install mne numpy scipy matplotlib pandas openpyxl

Usage:
    1. Fill in PARTICIPANT_METADATA below with your physical list info.
    2. Edit CONFIG if needed.
    3. Run: python eeg_multi_testing.py
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats
import json
import gc
from itertools import combinations

# =============================================================================
# PARTICIPANT METADATA
# =============================================================================
# Metadata is now loaded dynamically from the behavioral CSV instead of a
# hardcoded dict. The CSV must have (at minimum) these columns:
#   Participant_number : integer participant ID (matches the number in the
#                         EEG filename, e.g. 1 -> "01rec1.vhdr")
#   Condition           : one of "control" | "SWA-ENG" | "ENG-SWA"
#   Stress              : raw numeric stress score (continuous covariate -
#                         no more low/moderate/high binning)
#   Recall1             : accuracy on recall 1 (may be blank/NaN)
#   Recall2             : accuracy on recall 2 (may be blank/NaN)
#
# The key in PARTICIPANT_METADATA is the zero-padded participant number as a
# string (e.g. "01", "02", ... "70") to match existing filename conventions.
# Rows with a missing Condition are skipped entirely (can't be grouped).
# Missing Stress/Recall1/Recall2 values are kept as None and are simply
# skipped wherever they would break a given statistical test.
# -----------------------------------------------------------------------------

DATA_DIR       = Path(r"E:\Thesis folder UM\Analysis\Valid")
ALL_FILES      = sorted(set(DATA_DIR.glob("*rec*.vhdr")))
OUTPUT_ROOT    = Path(r"D:\Code\EEG_output")
BEHAVIORAL_CSV = Path(r"E:\Thesis folder UM\Analysis\Participants_Behavioral_data.csv")


def _clean_col(name):
    """Normalize a CSV column header: strip whitespace/BOM."""
    return str(name).strip().lstrip("\ufeff")


def load_participant_metadata(csv_path=BEHAVIORAL_CSV):
    """
    Build the PARTICIPANT_METADATA dict from the behavioral CSV.

    Returns dict: {"01": {"condition": ..., "stress": <float|None>,
                           "accuracy_recall1": <float|None>,
                           "accuracy_recall2": <float|None>}, ...}
    """
    df = pd.read_csv(csv_path)
    df.columns = [_clean_col(c) for c in df.columns]

    # Column name mapping - tolerate the trailing space on "Recall1 " and any
    # stray unnamed columns from blank Encoding cells in the raw sheet.
    colmap = {}
    for c in df.columns:
        key = c.strip().lower()
        if key == "participant_number":
            colmap["participant_number"] = c
        elif key == "condition":
            colmap["condition"] = c
        elif key == "stress":
            colmap["stress"] = c
        elif key == "recall1":
            colmap["recall1"] = c
        elif key == "recall2":
            colmap["recall2"] = c

    required = ["participant_number", "condition", "stress", "recall1", "recall2"]
    missing = [r for r in required if r not in colmap]
    if missing:
        raise ValueError(f"Behavioral CSV is missing required column(s): {missing}. "
                          f"Found columns: {list(df.columns)}")

    def _num_or_none(val):
        return float(val) if pd.notna(val) else None

    metadata = {}
    for _, row in df.iterrows():
        raw_num = row[colmap["participant_number"]]
        if pd.isna(raw_num):
            continue
        num = str(int(raw_num)).zfill(2)

        condition = row[colmap["condition"]]
        if pd.isna(condition):
            print(f"    WARNING: Participant {num} has no Condition - skipping.")
            continue
        condition = str(condition).strip()

        metadata[num] = {
            "condition": condition,
            "stress": _num_or_none(row[colmap["stress"]]),
            "accuracy_recall1": _num_or_none(row[colmap["recall1"]]),
            "accuracy_recall2": _num_or_none(row[colmap["recall2"]]),
        }

    print(f"    Loaded metadata for {len(metadata)} participants from {csv_path}")
    return metadata


PARTICIPANT_METADATA = load_participant_metadata()

# Valid labels (edit these if you rename your groups)
CONDITIONS = ["control", "SWA-ENG", "ENG-SWA"]

# =============================================================================
# CONFIG - edit these values to match your data
# =============================================================================


# --- Re-referencing ---
REF2_CHANNEL = "ref2"

# --- Filtering ---
HIGHPASS_HZ  = 0.5
LOWPASS_HZ   = 30.0
FILTER_ORDER = 2

# --- Ocular correction ---
EOG_CHANNELS = ["HEOG left", "HEOG right", "VEOG above", "VEOG below"]
CHANNELS_FOR_OCULAR_CORRECTION = None

# --- Segmentation ---
EPOCH_MARKER = "Stimulus/S  1"
EPOCH_TMIN   = -0.200
EPOCH_TMAX   =  3.000
BASELINE     = (-0.200, 0.0)

# --- Artifact rejection ---
ARTIFACT_CHANNELS = ["Fz", "FCz", "Cz", "CPz", "Pz"]
ARTIFACT_AMP_MIN  = -100e-6
ARTIFACT_AMP_MAX  =  100e-6
ARTIFACT_TIME_WIN = (-0.200, 0.200)

# --- Wavelet (CWT Morlet) ---
WAVELET_FREQS   = np.arange(4, 31, 1)
MORLET_PARAM    = WAVELET_FREQS / 2
OUTPUT_AMPLITUDE = True
OUTPUT_POWER     = True

# --- Frequency bands ---
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}

# --- Topomap / time-course settings ---
TOPOMAP_TIMES = [-0.1, 0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9]
KEY_CHANNELS  = ["Fz", "FCz", "Cz", "CPz", "Pz", "Oz"]

# Frontal-midline subset used specifically for the H3 stress/theta/memory analysis.
# Frontal midline theta (FM-theta) is the channel-specific signature reported in the
# memory/stress literature (Fz, FCz are the canonical sites). Averaging across the
# full KEY_CHANNELS list (which includes parietal/occipital Pz, Oz) dilutes this
# effect - those sites aren't part of the theoretical claim in H3.
FRONTAL_CHANNELS = ["Fz", "FCz"]

# Time-windows for the summary tables (seconds)
# Each entry: (label, tmin, tmax)
TABLE_TIME_WINDOWS = [
    ("baseline",  -0.200,  0.000),
    ("early",      0.000,  0.500),
    ("mid",        0.500,  1.500),
    ("late",       1.500,  3.000),
]

# --- Skip flags ---
SKIP_REREF  = False
SKIP_FILTER = False
SKIP_OCULAR = False

# =============================================================================
# HELPERS - participant lookup
# =============================================================================

def get_participant_num(filepath):
    """
    Extract the participant number from various filename formats:
      - '07rec1.vhdr'          -> '07'   (leading digits)
      - 'StuStra3_52_rec1.vhdr'-> '52'   (last _NUMBER_ before rec/wprec)
      - 'WPrec1_01.vhdr'       -> '01'   (trailing _NUMBER)
    Returns the number string as it appears (with leading zeros), or None.
    """
    import re
    stem = Path(filepath).stem  # strip extension

    # Pattern 1: leading digits  e.g. "07rec1"
    m = re.match(r"^(\d+)", stem)
    if m:
        return m.group(1)

    # Pattern 2: _NUMBER_ followed by rec/wprec  e.g. "StuStra3_52_rec1"
    m = re.search(r"_(\d+)_(?:rec|wprec)", stem, re.IGNORECASE)
    if m:
        return m.group(1)

    # Pattern 3: trailing _NUMBER  e.g. "WPrec1_01"
    m = re.search(r"_(\d+)$", stem)
    if m:
        return m.group(1)

    return None


def get_metadata(filepath):
    """Return {'stress': ..., 'condition': ...} for a file, or None if not found."""
    num = get_participant_num(filepath)
    if num is None:
        print(f"    WARNING: Could not extract participant number from '{Path(filepath).name}'")
        return None
    # Try both zero-padded and non-padded
    meta = PARTICIPANT_METADATA.get(num) or PARTICIPANT_METADATA.get(num.lstrip("0") or "0")
    if meta is None:
        print(f"    WARNING: Participant '{num}' not in PARTICIPANT_METADATA - file will be skipped in group analysis.")
    return meta


def group_by_recall(files):
    recall1, recall2 = [], []
    for f in files:
        name = f.name.lower()
        if "recall1" in name or "rec1" in name or "wprec1" in name:
            recall1.append(f)
        elif "recall2" in name or "rec2" in name or "wprec2" in name:
            recall2.append(f)
        else:
            print(f"    WARNING: '{f.name}' does not match recall1/recall2 - skipping.")
    return recall1, recall2


# =============================================================================
# PIPELINE (processing - unchanged from original)
# =============================================================================

OUTPUT_DIR = None   # set per-file inside run_pipeline()


def load_raw(filepath):
    print(f"\n[1] Loading: {filepath}")
    raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
    print(f"    Channels : {len(raw.ch_names)}")
    print(f"    Duration : {raw.times[-1]:.1f} s")
    print(f"    Sfreq    : {raw.info['sfreq']} Hz")
    raw.rename_channels({"FP1": "Fp1", "FP2": "Fp2"})
    print("\nCHANNEL NAMES:")
    print(raw.ch_names)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    print(raw.get_montage())
    print(raw.info["chs"][0]["loc"][:3])
    return raw


def set_reference(raw):
    print("\n[2] Re-referencing to linked mastoids")
    raw = mne.add_reference_channels(raw, ['Ref1'])
    raw_reref, _ = mne.set_eeg_reference(raw, ref_channels=['Ref1', 'ref2'], copy=True)
    return raw_reref


def apply_filter(raw):
    print(f"\n[3] Filtering: {HIGHPASS_HZ}-{LOWPASS_HZ} Hz  (order {FILTER_ORDER})")
    iir_params = dict(order=FILTER_ORDER, ftype="butter", output="sos")
    raw_filt = raw.copy().filter(
        l_freq=HIGHPASS_HZ, h_freq=LOWPASS_HZ,
        method="iir", iir_params=iir_params, verbose=False,
    )
    return raw_filt


def ocular_correction(raw):
    print(f"\n[4] Ocular correction using: {EOG_CHANNELS}")
    if CHANNELS_FOR_OCULAR_CORRECTION is None:
        exclude    = set(EOG_CHANNELS + [REF2_CHANNEL])
        target_chs = [ch for ch in raw.ch_names if ch not in exclude]
    else:
        target_chs = CHANNELS_FOR_OCULAR_CORRECTION

    data     = raw.get_data()
    ch_names = raw.ch_names
    eog_idx  = [ch_names.index(ch) for ch in EOG_CHANNELS]
    eog_data = data[eog_idx, :].T
    X        = np.column_stack([eog_data, np.ones(eog_data.shape[0])])
    raw_corr = raw.copy()
    corr_data = raw_corr.get_data()

    for ch in target_chs:
        idx = ch_names.index(ch)
        y   = data[idx, :]
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        corr_data[idx, :] = y - X[:, :-1] @ coeffs[:-1]

    raw_corr._data = corr_data
    print(f"    Corrected {len(target_chs)} channels.")
    return raw_corr


def segment_epochs(raw):
    print(f"\n[5] Segmentation: marker='{EPOCH_MARKER}'  tmin={EPOCH_TMIN}s  tmax={EPOCH_TMAX}s")
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    print(f"    All markers found: {event_id}")
    if EPOCH_MARKER not in event_id:
        raise ValueError(
            f"Marker '{EPOCH_MARKER}' not found.\n"
            f"Available markers: {list(event_id.keys())}\n"
            f"Edit EPOCH_MARKER in CONFIG to match exactly."
        )
    epochs = mne.Epochs(
        raw, events,
        event_id={EPOCH_MARKER: event_id[EPOCH_MARKER]},
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=BASELINE, preload=True, verbose=False,
    )
    print(f"    Created {len(epochs)} epochs.")
    return epochs


def artifact_rejection(epochs):
    print(f"\n[6] Artifact rejection on: {ARTIFACT_CHANNELS}  "
          f"range=[{ARTIFACT_AMP_MIN*1e6:.0f}, {ARTIFACT_AMP_MAX*1e6:.0f}] uV  "
          f"window={ARTIFACT_TIME_WIN}s")
    times     = epochs.times
    tmin, tmax = ARTIFACT_TIME_WIN
    time_mask  = (times >= tmin) & (times <= tmax)
    ch_indices = [epochs.ch_names.index(ch) for ch in ARTIFACT_CHANNELS if ch in epochs.ch_names]
    if not ch_indices:
        print("    WARNING: None of the artifact channels found - skipping rejection.")
        return epochs
    data       = epochs.get_data()
    bad_epochs = []
    for ep_idx in range(len(epochs)):
        window_data = data[ep_idx][np.ix_(ch_indices, time_mask)]
        if window_data.max() > ARTIFACT_AMP_MAX or window_data.min() < ARTIFACT_AMP_MIN:
            bad_epochs.append(ep_idx)
    print(f"    Bad epochs: {len(bad_epochs)} / {len(epochs)}")
    epochs = epochs.drop(bad_epochs, reason="amplitude_artifact")
    print(f"    Remaining epochs: {len(epochs)}")
    return epochs


def compute_wavelets(epochs):
    print(f"\n[7] CWT Morlet  freqs={np.round(WAVELET_FREQS, 2)} Hz  n_cycles={MORLET_PARAM}")
    tfr = mne.time_frequency.tfr_morlet(
        epochs, freqs=WAVELET_FREQS, n_cycles=MORLET_PARAM,
        use_fft=True, return_itc=False, average=False, output="complex", verbose=False,
    )
    norm_factor = np.sqrt(WAVELET_FREQS)[np.newaxis, np.newaxis, :, np.newaxis]
    complex_data = tfr.data
    amplitude    = np.abs(complex_data) / norm_factor
    power        = amplitude ** 2
    print(f"    TFR data shape: {amplitude.shape}  (epochs x channels x freqs x times)")
    return tfr, amplitude, power


def extract_bands(amplitude, epochs):
    freqs     = WAVELET_FREQS
    band_data = {}
    for band_name, (fmin, fmax) in BANDS.items():
        freq_mask = (freqs >= fmin) & (freqs < fmax)
        if not freq_mask.any():
            print(f"    WARNING: No frequencies in band '{band_name}' ({fmin}-{fmax} Hz)")
            continue
        band_amp = amplitude[:, :, freq_mask, :].mean(axis=2)
        band_data[band_name] = band_amp
        print(f"    Band '{band_name}': {fmin}-{fmax} Hz -> shape {band_amp.shape}")
    return band_data


def save_results(epochs, tfr, amplitude, power, band_data=None):
    OUTPUT_DIR.mkdir(exist_ok=True)
    ep_path = OUTPUT_DIR / "epochs_clean-epo.fif"
    epochs.save(ep_path, overwrite=True)
    print(f"\n[8] Saved epochs -> {ep_path}")
    if OUTPUT_AMPLITUDE:
        np.save(OUTPUT_DIR / "cwt_amplitude.npy", amplitude)
    if OUTPUT_POWER:
        np.save(OUTPUT_DIR / "cwt_power.npy", power)
    if band_data:
        np.savez(OUTPUT_DIR / "band_data.npz", **{k: v for k, v in band_data.items()})
    np.savez(OUTPUT_DIR / "cwt_metadata.npz",
             ch_names=np.array(epochs.ch_names),
             freqs=WAVELET_FREQS,
             times=epochs.times,
             sfreq=np.array([epochs.info["sfreq"]]))
    print(f"    Saved metadata -> {OUTPUT_DIR / 'cwt_metadata.npz'}")
    return amplitude, power


def run_pipeline(file_path, output_root=None):
    global OUTPUT_DIR
    if output_root is not None:
        OUTPUT_DIR = output_root / Path(file_path).stem
    else:
        OUTPUT_DIR = OUTPUT_ROOT / Path(file_path).stem
    raw     = load_raw(file_path)
    raw     = set_reference(raw)
    raw     = apply_filter(raw)
    raw     = ocular_correction(raw)
    epochs  = segment_epochs(raw)
    epochs  = artifact_rejection(epochs)
    tfr, amplitude, power = compute_wavelets(epochs)
    band_data = extract_bands(amplitude, epochs)
    save_results(epochs, tfr, amplitude, power, band_data)
    print(f"\nOK Finished {Path(file_path).stem}\n")
    return band_data


# =============================================================================
# ANALYSIS - grand averages, topomaps, time courses, TF TABLES
# =============================================================================

def compute_grand_averages(band_data):
    """
    band_data: dict  band_name -> list of arrays, each either:
                 (n_epochs, n_channels, n_times)  - raw per-subject band data
               OR
                 (n_channels, n_times)             - already epoch-averaged
    Returns:   dict  band_name -> {'mean', 'sem', 'per_subject'}
    """
    grand_averages = {}
    for band_name, per_subject_data in band_data.items():
        if not per_subject_data:
            print(f"    WARNING: No data for band '{band_name}' - skipping.")
            continue

        subj_means = []
        for d in per_subject_data:
            arr = np.array(d)
            if arr.ndim == 3:
                # (n_epochs, n_channels, n_times) -> average over epochs
                subj_means.append(arr.mean(axis=0))
            elif arr.ndim == 2:
                # already (n_channels, n_times)
                subj_means.append(arr)
            else:
                print(f"    WARNING: Unexpected array shape {arr.shape} for band '{band_name}' - skipping subject.")
                continue

        if not subj_means:
            continue

        stacked = np.stack(subj_means, axis=0)   # (n_subjects, n_channels, n_times)
        n       = stacked.shape[0]
        mean    = stacked.mean(axis=0)
        sem     = stacked.std(axis=0) / np.sqrt(n) if n > 1 else np.zeros_like(mean)

        grand_averages[band_name] = {
            'mean':        mean,
            'sem':         sem,
            'per_subject': stacked,
        }
        print(f"    Grand-average '{band_name}': {n} subject(s), shape {mean.shape}")
    return grand_averages


def _get_eeg_picks_and_info(epochs):
    EEG_CHANNELS = [
        'Fz','FCz','Cz','CPz','Pz','Oz',
        'Fp1','Fp2','F3','F4',
        'FC3','FC4','C3','C4',
        'CP3','CP4','P3','P4',
        'O1','O2','F7','F8','FT7','FT8','T7','T8','TP7'
    ]
    eeg_picks = mne.pick_channels(epochs.ch_names, EEG_CHANNELS)
    info      = mne.pick_info(epochs.info, eeg_picks)
    return eeg_picks, info


def plot_topomaps_per_band(grand_averages, epochs, band_name, group_label, save_dir):
    """
    Topomaps for one band at TOPOMAP_TIMES, saved to save_dir.
    group_label is a string like 'condition_A' or 'stress_high'.

    Fixes applied (previously caused misleading/blobby maps):
      1. ONE fixed color scale for the whole band (computed from all time
         points), not a new vmin/vmax per frame. Per-frame rescaling made
         tiny differences look like huge changes and made frames
         incomparable to each other.
      2. Sequential colormap ('viridis') instead of diverging ('RdBu_r').
         This is amplitude/power data, which is non-negative by
         construction (it's an abs() of a complex wavelet coefficient) -
         a diverging red/blue map implies a meaningful zero-crossing that
         isn't actually there.
      3. extrapolate='local' to reduce the wedge-shaped artifacts at the
         scalp edge that come from sparse channel coverage (27 channels).
      4. Title now reports values in the same units already used
         elsewhere (raw volts are tiny, e.g. 1e-6, which displayed as a
         misleading "0.0000" at 4 decimal places) - switched to
         scientific notation so the real magnitude is visible.
    """
    times             = epochs.times
    eeg_picks, info   = _get_eeg_picks_and_info(epochs)

    # --- Fix 1 & 2: one global, non-negative-aware scale for the whole band ---
    all_band_data = grand_averages[band_name]['mean'][eeg_picks, :]
    data_is_nonnegative = bool(all_band_data.min() >= 0)
    absmax = float(np.percentile(np.abs(all_band_data), 99))
    if absmax == 0:
        absmax = float(np.abs(all_band_data).max()) or 1e-12

    if data_is_nonnegative:
        vlim = (0.0, absmax)
        cmap = "viridis"
    else:
        vlim = (-absmax, absmax)
        cmap = "RdBu_r"

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes      = axes.ravel()

    for ax_idx in range(min(len(TOPOMAP_TIMES), 9)):
        target_time = TOPOMAP_TIMES[ax_idx]
        time_idx    = np.argmin(np.abs(times - target_time))
        time_val    = times[time_idx]
        data_mean   = grand_averages[band_name]['mean'][eeg_picks, time_idx]
        data_sem    = grand_averages[band_name]['sem'][eeg_picks, time_idx]

        mne.viz.plot_topomap(
            data_mean, info, axes=axes[ax_idx],
            vlim=vlim, cmap=cmap, contours=6, sphere="auto",
            extrapolate="head", show=False,
        )
        axes[ax_idx].set_title(
            f'{band_name.upper()} @ {time_val:.2f}s\n'
            f'mean={data_mean.mean():.3e}  +/-SEM={data_sem.mean():.3e}'
        )

    plt.suptitle(f'Grand-Average {band_name.upper()} - {group_label}', fontsize=14)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / f'topomaps_{band_name}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved topomaps -> {plot_path}")


def plot_band_time_courses(grand_averages, epochs, group_label, save_dir):
    """Time-course plots (mean +/- SEM) for each band x key channel."""
    times = epochs.times
    fig, axes = plt.subplots(len(BANDS), len(KEY_CHANNELS), figsize=(20, 3 * len(BANDS)))

    for band_idx, (band_name, _) in enumerate(BANDS.items()):
        grand = grand_averages[band_name]
        mean  = grand['mean']
        sem   = grand['sem']
        for ch_plot_idx, ch_name in enumerate(KEY_CHANNELS):
            if ch_name not in epochs.ch_names:
                continue
            ch_data_idx = epochs.ch_names.index(ch_name)
            ax = axes[band_idx, ch_plot_idx]
            ax.plot(times, mean[ch_data_idx, :], 'b-', linewidth=2, label='Mean')
            ax.fill_between(
                times,
                mean[ch_data_idx, :] - sem[ch_data_idx, :],
                mean[ch_data_idx, :] + sem[ch_data_idx, :],
                alpha=0.3, color='blue', label='+/-SEM'
            )
            ax.axvline(0, color='red', linestyle='--', linewidth=1, label='S1')
            ax.set_title(f'{band_name.upper()} - {ch_name}')
            ax.set_xlabel('Time (s)')
            if ch_plot_idx == 0:
                ax.set_ylabel('Amplitude')
            if band_idx == 0 and ch_plot_idx == 0:
                ax.legend(fontsize=8)

    plt.suptitle(f'Band Time Courses - {group_label}', fontsize=14)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / 'time_courses.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved time courses -> {plot_path}")


# =============================================================================
# TIME-FREQUENCY SUMMARY TABLES
# =============================================================================

def build_tf_table(grand_averages, epochs, group_label):
    """
    Build a tidy DataFrame with mean amplitude per:
      band x time-window x channel (KEY_CHANNELS only)

    Columns:
      group | band | time_window | t_start | t_end | channel | mean_amplitude | sem_amplitude

    Returns a pandas DataFrame.
    """
    times = epochs.times
    rows  = []

    for band_name, (fmin, fmax) in BANDS.items():
        grand = grand_averages[band_name]
        mean  = grand['mean']   # (n_channels, n_times)
        sem   = grand['sem']

        for win_label, tmin, tmax in TABLE_TIME_WINDOWS:
            win_mask = (times >= tmin) & (times <= tmax)
            if not win_mask.any():
                continue

            for ch_name in KEY_CHANNELS:
                if ch_name not in epochs.ch_names:
                    continue
                ch_idx   = epochs.ch_names.index(ch_name)
                amp_mean = mean[ch_idx, win_mask].mean()
                amp_sem  = sem[ch_idx, win_mask].mean()

                rows.append({
                    "group":         group_label,
                    "band":          band_name,
                    "freq_range_Hz": f"{fmin}-{fmax}",
                    "time_window":   win_label,
                    "t_start_s":     tmin,
                    "t_end_s":       tmax,
                    "channel":       ch_name,
                    "mean_amplitude": round(float(amp_mean), 6),
                    "sem_amplitude":  round(float(amp_sem),  6),
                })

    df = pd.DataFrame(rows)
    return df


def build_tf_table_allchannels(grand_averages, epochs, group_label):
    """
    Like build_tf_table but uses ALL EEG channels (not just KEY_CHANNELS).
    Useful for topographic comparisons.
    """
    times = epochs.times
    eeg_picks, _ = _get_eeg_picks_and_info(epochs)
    eeg_ch_names = [epochs.ch_names[i] for i in eeg_picks]
    rows = []

    for band_name, (fmin, fmax) in BANDS.items():
        grand = grand_averages[band_name]
        mean  = grand['mean']
        sem   = grand['sem']

        for win_label, tmin, tmax in TABLE_TIME_WINDOWS:
            win_mask = (times >= tmin) & (times <= tmax)
            if not win_mask.any():
                continue

            for ch_name in eeg_ch_names:
                if ch_name not in epochs.ch_names:
                    continue
                ch_idx   = epochs.ch_names.index(ch_name)
                amp_mean = mean[ch_idx, win_mask].mean()
                amp_sem  = sem[ch_idx, win_mask].mean()

                rows.append({
                    "group":          group_label,
                    "band":           band_name,
                    "freq_range_Hz":  f"{fmin}-{fmax}",
                    "time_window":    win_label,
                    "t_start_s":      tmin,
                    "t_end_s":        tmax,
                    "channel":        ch_name,
                    "mean_amplitude": round(float(amp_mean), 6),
                    "sem_amplitude":  round(float(amp_sem),  6),
                })

    return pd.DataFrame(rows)


def save_tf_tables(all_dfs, save_dir, filename_stem):
    """
    Concatenate a list of DataFrames and save as CSV + Excel.
    Each DataFrame should already have a 'group' column.
    """
    if not all_dfs:
        print("    No TF table data to save.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = save_dir / f"{filename_stem}.csv"
    xlsx_path = save_dir / f"{filename_stem}.xlsx"

    combined.to_csv(csv_path, index=False)
    print(f"    Saved TF table (CSV)  -> {csv_path}")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="all_groups", index=False)
        # Also one sheet per group
        for grp, sub_df in combined.groupby("group"):
            sheet_name = str(grp)[:31]  # Excel sheet name limit
            sub_df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"    Saved TF table (XLSX) -> {xlsx_path}")

    return combined


# =============================================================================
# GROUPING LOGIC - split accumulated data by condition / stress
# =============================================================================

def run_analysis_for_groups(group_accum_by_label, recall_name, rep_epochs_store, dimension):
    """
    group_accum_by_label : dict  label -> {band_name: [subj_mean_arr, ...]}
                           label can be a string (condition/stress) or a tuple (condition, stress)
    recall_name          : e.g. "recall1"
    rep_epochs_store     : dict  label -> mne.Epochs  (representative epochs object)
    dimension            : "condition" (stress is now a continuous covariate,
                            so stress-group and combined topomaps are no longer
                            produced here - see H3/H3b for stress analyses)
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS BY {dimension.upper()} - {recall_name}")
    print("="*60)

    all_tf_dfs = []

    for label, band_data in group_accum_by_label.items():
        if not any(band_data.values()):
            print(f"    No data for {dimension}={label} - skipping.")
            continue

        rep_epochs = rep_epochs_store.get(label)
        if rep_epochs is None:
            print(f"    WARNING: No representative epochs for {label} - skipping plots.")
            continue

        # Convert tuple labels to readable strings, e.g. ("control", "low") -> "control_low"
        if isinstance(label, tuple):
            label_str = "_".join(str(x) for x in label)
        else:
            label_str = str(label)

        grand_averages = compute_grand_averages(band_data)
        group_label    = f"{recall_name}_{dimension}_{label_str}"
        save_dir       = OUTPUT_ROOT / recall_name / dimension / label_str

        # -- topomaps ----------------------------------------------------------
        for band_name in BANDS:
            plot_topomaps_per_band(grand_averages, rep_epochs, band_name, group_label, save_dir)

        # -- time courses ------------------------------------------------------
        plot_band_time_courses(grand_averages, rep_epochs, group_label, save_dir)

        # -- TF table (key channels) -------------------------------------------
        df = build_tf_table(grand_averages, rep_epochs, group_label)
        all_tf_dfs.append(df)

        # -- TF table (all channels) -------------------------------------------
        df_all = build_tf_table_allchannels(grand_averages, rep_epochs, group_label)
        save_tf_tables([df_all], save_dir, f"tf_table_{group_label}_all_channels")

        # Release this group's grand-average arrays (each holds a full
        # per-subject stack per band) and any stray open figures before
        # moving to the next condition.
        del grand_averages
        plt.close('all')
        gc.collect()

    # Save combined key-channel table for this recall x dimension
    combined_dir = OUTPUT_ROOT / recall_name / dimension
    save_tf_tables(all_tf_dfs, combined_dir, f"tf_table_{recall_name}_{dimension}_keychannels")


# =============================================================================
# BEHAVIORAL-EEG ANALYSIS
# =============================================================================

def _load_subject_band_data(recall_dir):
    """
    Walk saved subject directories and load band_data.npz + cwt_metadata.npz.
    Returns dict {subject_num: {condition, stress, accuracy, band_data: {band: (n_ch, n_time)}, ch_names, times}}
    """
    subject_bands = {}
    if not recall_dir.exists():
        print(f"    WARNING: {recall_dir} does not exist.")
        return subject_bands

    for subj_dir in sorted(recall_dir.iterdir()):
        if not subj_dir.is_dir():
            continue

        band_path = subj_dir / "band_data.npz"
        meta_path = subj_dir / "cwt_metadata.npz"

        if not band_path.exists() or not meta_path.exists():
            continue

        subj_num = get_participant_num(subj_dir)
        if subj_num is None:
            continue

        meta = PARTICIPANT_METADATA.get(subj_num) or PARTICIPANT_METADATA.get(subj_num.lstrip("0") or "0")
        if meta is None or meta.get("accuracy_recall2") is None:
            continue

        try:
            band_data = np.load(band_path, allow_pickle=True)
            meta_data = np.load(meta_path, allow_pickle=True)

            times = meta_data["times"]
            ch_names = list(meta_data["ch_names"])

            loaded_bands = {}
            for band_name in BANDS:
                if band_name in band_data:
                    arr = np.array(band_data[band_name])
                    if arr.ndim == 3:
                        loaded_bands[band_name] = arr.mean(axis=0)  # avg over epochs
                    else:
                        loaded_bands[band_name] = arr

            if loaded_bands:
                subject_bands[subj_num] = {
                    "condition": meta["condition"],
                    "stress": meta["stress"],
                    "accuracy_recall2": meta["accuracy_recall2"],
                    "band_data": loaded_bands,
                    "ch_names": ch_names,
                    "times": times,
                }
        except Exception as e:
            print(f"    WARNING: Failed to load {band_path}: {e}")

    print(f"    Loaded {len(subject_bands)} subjects from {recall_dir}")
    return subject_bands


def describe_accuracy_by_group(subject_bands, dimension="condition"):
    """Compute mean / SEM / min / max accuracy per group.

    Note: stress is now a continuous covariate (see H3/H3b), so the only
    supported grouping dimension is "condition".
    """
    rows = []

    if dimension != "condition":
        raise ValueError(f"Unsupported dimension '{dimension}' - only 'condition' "
                          f"grouping is supported now that stress is continuous.")
    groups = CONDITIONS
    group_key = lambda s: s["condition"]

    for grp in groups:
        accs = [s["accuracy_recall2"] for s in subject_bands.values() if group_key(s) == grp]
        if not accs:
            continue
        rows.append({
            "dimension": dimension,
            "group": grp,
            "n": len(accs),
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "sem_accuracy": round(float(scipy_stats.sem(accs)), 4) if len(accs) > 1 else 0.0,
            "min_accuracy": round(float(np.min(accs)), 4),
            "max_accuracy": round(float(np.max(accs)), 4),
        })

    return pd.DataFrame(rows)


def compare_accuracy_groups(subject_bands, dimension="condition"):
    """Kruskal-Wallis across groups, pairwise Mann-Whitney U post-hoc with Bonferroni correction.

    Note: stress is now a continuous covariate, so only "condition" grouping is supported.
    """
    rows = []

    if dimension != "condition":
        raise ValueError(f"Unsupported dimension '{dimension}' - only 'condition' "
                          f"grouping is supported now that stress is continuous.")
    group_key = lambda s: s["condition"]

    # Collect accuracy values per group
    group_data = {}
    for s in subject_bands.values():
        grp = group_key(s)
        group_data.setdefault(grp, []).append(s["accuracy_recall2"])

    groups = list(group_data.keys())
    if len(groups) < 3:
        print(f"    WARNING: Need >= 3 groups for Kruskal-Wallis, got {len(groups)}")
        return pd.DataFrame(rows)

    # Kruskal-Wallis
    all_accs = [v for v in group_data.values()]
    kw_stat, kw_p = scipy_stats.kruskal(*all_accs)
    rows.append({
        "test": "Kruskal-Wallis",
        "dimension": dimension,
        "n_groups": len(groups),
        "statistic": round(float(kw_stat), 4),
        "p_value": round(float(kw_p), 6),
        "significant": bool(kw_p < 0.05),
    })

    # Post-hoc pairwise Mann-Whitney U with Bonferroni
    if kw_p < 0.05:
        from itertools import combinations
        n_tests = len(list(combinations(groups, 2)))
        for g1, g2 in combinations(groups, 2):
            u_stat, u_p = scipy_stats.mannwhitneyu(group_data[g1], group_data[g2], alternative='two-sided')
            rows.append({
                "test": f"Mann-Whitney U ({g1} vs {g2})",
                "dimension": dimension,
                "group1": g1,
                "group2": g2,
                "U_statistic": round(float(u_stat), 4),
                "p_value_raw": round(float(u_p), 6),
                "p_value_corrected": round(min(float(u_p * n_tests), 1.0), 6),
                "significant": bool(u_p * n_tests < 0.05),
            })

    return pd.DataFrame(rows)


def compute_accuracy_eeg_correlations(subject_bands, dimension):
    """
    For each band x time-window x channel, compute Pearson r between
    accuracy and mean EEG amplitude across subjects in each subgroup.
    """
    rows = []

    if dimension != "condition":
        raise ValueError(f"Unsupported dimension '{dimension}' - only 'condition' "
                          f"grouping is supported now that stress is continuous.")
    group_key = lambda s: s["condition"]

    # Group subjects
    groups = {}
    for subj_num, s in subject_bands.items():
        grp = group_key(s)
        groups.setdefault(grp, []).append(s)

    for grp, members in groups.items():
        if len(members) < 3:
            print(f"    WARNING: Only {len(members)} subjects in {grp} - skipping correlation.")
            continue

        accs = np.array([m["accuracy_recall2"] for m in members])

        for band_name, (fmin, fmax) in BANDS.items():
            times = members[0]["times"]

            for win_label, tmin, tmax in TABLE_TIME_WINDOWS:
                win_mask = (times >= tmin) & (times <= tmax)
                if not win_mask.any():
                    continue

                for ch_idx, ch_name in enumerate(members[0]["ch_names"]):
                    if ch_name not in KEY_CHANNELS:
                        continue

                    subj_amplitudes = []
                    for m in members:
                        amp = m["band_data"][band_name][ch_idx, win_mask].mean()
                        subj_amplitudes.append(amp)
                    subj_amplitudes = np.array(subj_amplitudes)

                    if np.std(accs) > 0 and np.std(subj_amplitudes) > 0:
                        r, p = scipy_stats.pearsonr(accs, subj_amplitudes)
                    else:
                        r, p = 0.0, 1.0

                    rows.append({
                        "dimension": dimension,
                        "group": grp,
                        "band": band_name,
                        "freq_range_Hz": f"{fmin}-{fmax}",
                        "time_window": win_label,
                        "channel": ch_name,
                        "r_value": round(float(r), 4),
                        "p_value": round(float(p), 6),
                        "n_subjects": len(members),
                        "significant": bool(p < 0.05),
                    })

    return pd.DataFrame(rows)


def plot_accuracy_vs_eeg_scatter(subject_bands, dimension, save_dir):
    """Scatter plots: accuracy vs mean EEG amplitude per band, one subplot per group."""
    save_dir.mkdir(parents=True, exist_ok=True)

    if dimension != "condition":
        raise ValueError(f"Unsupported dimension '{dimension}' - only 'condition' "
                          f"grouping is supported now that stress is continuous.")
    group_key = lambda s: s["condition"]

    groups = {}
    for subj_num, s in subject_bands.items():
        grp = group_key(s)
        groups.setdefault(grp, []).append(s)

    for grp, members in groups.items():
        if len(members) < 2:
            continue

        fig, axes = plt.subplots(1, len(BANDS), figsize=(5 * len(BANDS), 5))
        if len(BANDS) == 1:
            axes = [axes]

        for band_idx, (band_name, (fmin, fmax)) in enumerate(BANDS.items()):
            ax = axes[band_idx]
            accs = np.array([m["accuracy_recall2"] for m in members])

            subj_amplitudes = []
            for m in members:
                bdata = m["band_data"][band_name]
                key_idx = [i for i, ch in enumerate(m["ch_names"]) if ch in KEY_CHANNELS]
                amp = bdata[key_idx, :].mean() if key_idx else np.nan
                subj_amplitudes.append(amp)
            subj_amplitudes = np.array(subj_amplitudes)

            ax.scatter(subj_amplitudes, accs, s=80, alpha=0.7, edgecolors='k', linewidth=0.5)

            valid = ~(np.isnan(subj_amplitudes))
            if valid.any() and np.std(subj_amplitudes[valid]) > 0:
                z = np.polyfit(subj_amplitudes[valid], accs[valid], 1)
                p = np.poly1d(z)
                x_line = np.linspace(subj_amplitudes[valid].min(), subj_amplitudes[valid].max(), 100)
                ax.plot(x_line, p(x_line), 'r--', linewidth=1.5)

                r, pval = scipy_stats.pearsonr(subj_amplitudes[valid], accs[valid])
                ax.text(0.05, 0.95, f'r={r:.3f}\np={pval:.4f}',
                       transform=ax.transAxes, va='top', fontsize=10)

            ax.set_xlabel(f'{band_name.upper()} amplitude ({fmin}-{fmax} Hz)')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{grp}')

        plt.suptitle(f'Accuracy vs EEG Amplitude - {dimension}')
        plt.tight_layout()

        label = grp if isinstance(grp, str) else '_'.join(str(x) for x in grp)
        plot_path = save_dir / f'scatter_{dimension}_{label}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved scatter -> {plot_path}")


def plot_accuracy_eeg_heatmap(subject_bands, dimension, save_dir):
    """Heatmap: accuracy-EEG Pearson r per band x channel, averaged across time windows."""
    save_dir.mkdir(parents=True, exist_ok=True)

    if dimension != "condition":
        raise ValueError(f"Unsupported dimension '{dimension}' - only 'condition' "
                          f"grouping is supported now that stress is continuous.")
    group_key = lambda s: s["condition"]

    groups = {}
    for subj_num, s in subject_bands.items():
        grp = group_key(s)
        groups.setdefault(grp, []).append(s)

    for grp, members in groups.items():
        if len(members) < 3:
            continue

        accs = np.array([m["accuracy_recall2"] for m in members])
        n_bands = len(BANDS)
        n_chs = len(KEY_CHANNELS)
        corr_matrix = np.zeros((n_bands, n_chs))
        p_matrix = np.zeros((n_bands, n_chs))

        for band_idx, (band_name, _) in enumerate(BANDS.items()):
            for ch_idx, ch_name in enumerate(KEY_CHANNELS):
                subj_amplitudes = []
                for m in members:
                    bdata = m["band_data"][band_name]
                    if ch_name in m["ch_names"]:
                        ci = m["ch_names"].index(ch_name)
                        subj_amplitudes.append(bdata[ci, :].mean())

                if len(subj_amplitudes) == len(members) and np.std(subj_amplitudes) > 0:
                    subj_amplitudes = np.array(subj_amplitudes)
                    r, p = scipy_stats.pearsonr(accs, subj_amplitudes)
                    corr_matrix[band_idx, ch_idx] = r
                    p_matrix[band_idx, ch_idx] = p

        label = grp if isinstance(grp, str) else '_'.join(str(x) for x in grp)

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(corr_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(n_chs))
        ax.set_xticklabels(KEY_CHANNELS)
        ax.set_yticks(range(n_bands))
        ax.set_yticklabels([b.upper() for b in BANDS.keys()])
        ax.set_xlabel('Channel')
        ax.set_ylabel('Band')
        ax.set_title(f'Accuracy-EEG Correlation ({dimension}={grp})')

        for i in range(n_bands):
            for j in range(n_chs):
                sig = '*' if p_matrix[i, j] < 0.05 else ''
                ax.text(j, i, f'{corr_matrix[i, j]:.2f}{sig}',
                       ha='center', va='center', fontsize=9,
                       color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

        plt.colorbar(im, ax=ax, label="Pearson r")
        plt.tight_layout()

        plot_path = save_dir / f'heatmap_{label}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved heatmap -> {plot_path}")


def plot_accuracy_and_eeg_bars(subject_bands, dimension, save_dir):
    """Bar chart: accuracy and mean EEG amplitude side-by-side per group."""
    save_dir.mkdir(parents=True, exist_ok=True)

    if dimension != "condition":
        raise ValueError(f"Unsupported dimension '{dimension}' - only 'condition' "
                          f"grouping is supported now that stress is continuous.")
    group_key = lambda s: s["condition"]
    group_order = CONDITIONS

    fig, axes = plt.subplots(1, len(BANDS), figsize=(5 * len(BANDS), 5))
    if len(BANDS) == 1:
        axes = [axes]

    for band_idx, (band_name, (fmin, fmax)) in enumerate(BANDS.items()):
        ax = axes[band_idx]
        n_groups = len(group_order)
        bar_width = 0.35
        x = np.arange(n_groups)

        accs_list = []
        eeg_list = []
        acc_err = []
        eeg_err = []

        for gi, grp in enumerate(group_order):
            members = [s for s in subject_bands.values() if group_key(s) == grp]
            if not members:
                accs_list.append(np.nan)
                eeg_list.append(np.nan)
                acc_err.append(0)
                eeg_err.append(0)
                continue

            accs = np.array([m["accuracy_recall2"] for m in members])
            accs_list.append(float(accs.mean()))
            acc_err.append(float(scipy_stats.sem(accs)) if len(accs) > 1 else 0)

            eeg_vals = []
            for m in members:
                bdata = m["band_data"][band_name]
                key_idx = [i for i, ch in enumerate(m["ch_names"]) if ch in KEY_CHANNELS]
                if key_idx:
                    eeg_vals.append(bdata[key_idx, :].mean())
            eeg_list.append(float(np.mean(eeg_vals)) if eeg_vals else np.nan)
            eeg_err.append(float(np.std(eeg_vals) / np.sqrt(len(eeg_vals))) if len(eeg_vals) > 1 else 0)

        ax.bar(x - bar_width / 2, accs_list, bar_width, yerr=acc_err, label='Accuracy',
               capsize=3, color='steelblue', alpha=0.8)
        ax.bar(x + bar_width / 2, eeg_list, bar_width, yerr=eeg_err, label=f'{band_name.upper()} Amp',
               capsize=3, color='coral', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in group_order], rotation=15, ha='right')
        ax.set_ylabel('Value')
        ax.set_title(f'{band_name.upper()}')
        ax.legend(fontsize=8)

    plt.suptitle(f'Accuracy vs EEG Amplitude by {dimension}')
    plt.tight_layout()

    plot_path = save_dir / f'bar_{dimension}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved bar chart -> {plot_path}")


def save_behavioral_results(desc_df, comp_df, corr_df, save_dir):
    """Save all behavioral analysis tables as CSV."""
    save_dir.mkdir(parents=True, exist_ok=True)

    if desc_df is not None and not desc_df.empty:
        desc_df.to_csv(save_dir / "accuracy_descriptive.csv", index=False)
        print(f"    Saved accuracy descriptive stats -> {save_dir / 'accuracy_descriptive.csv'}")

    if comp_df is not None and not comp_df.empty:
        comp_df.to_csv(save_dir / "accuracy_group_comparisons.csv", index=False)
        print(f"    Saved group comparisons -> {save_dir / 'accuracy_group_comparisons.csv'}")

    if corr_df is not None and not corr_df.empty:
        corr_df.to_csv(save_dir / "accuracy_eeg_correlations.csv", index=False)
        print(f"    Saved accuracy-EEG correlations -> {save_dir / 'accuracy_eeg_correlations.csv'}")


def regenerate_topomaps_from_saved(recall_dir, recall_name, output_root):
    """
    Rebuild topomaps (and the other group-level plots/tables) from already
    saved band_data.npz + epochs_clean-epo.fif files on disk, WITHOUT
    re-running the raw EEG processing pipeline (filtering/ocular
    correction/CWT etc). Use this after fixing plot_topomaps_per_band
    instead of uncommenting the full raw-reprocessing block in main.

    Groups subjects by condition only. Stress is now a continuous covariate
    (see H3/H3b), so it is no longer used to bucket subjects into topomap
    groups here.
    """
    print(f"\n{'='*60}")
    print(f"REGENERATING TOPOMAPS FROM SAVED DATA - {recall_name}")
    print("="*60)

    if not recall_dir.exists():
        print(f"    WARNING: {recall_dir} does not exist - nothing to regenerate.")
        return

    cond_accum      = {c: {b: [] for b in BANDS} for c in CONDITIONS}
    cond_rep_epochs = {}

    n_loaded = 0
    for subj_dir in sorted(recall_dir.iterdir()):
        if not subj_dir.is_dir():
            continue

        band_path = subj_dir / "band_data.npz"
        ep_path   = subj_dir / "epochs_clean-epo.fif"
        if not band_path.exists() or not ep_path.exists():
            continue

        meta = get_metadata(subj_dir)
        if meta is None:
            continue
        condition = meta["condition"]
        if condition not in CONDITIONS:
            continue

        try:
            bdata = np.load(band_path, allow_pickle=True)
            for band_name in BANDS:
                if band_name not in bdata:
                    continue
                band_arr = np.asarray(bdata[band_name], dtype=np.float32)
                if band_arr.ndim == 3:
                    # Average over epochs NOW instead of keeping the full
                    # (n_epochs, n_channels, n_times) array in memory for
                    # every subject at once - compute_grand_averages only
                    # ever needs the per-subject epoch-mean, so holding the
                    # raw epoch-level data for all ~60 subjects x 3 bands
                    # simultaneously is what was blowing up RAM.
                    band_arr = band_arr.mean(axis=0)
                cond_accum[condition][band_name].append(band_arr)
            bdata.close()

            if condition not in cond_rep_epochs:
                cond_rep_epochs[condition] = mne.read_epochs(ep_path, verbose=False)

            n_loaded += 1
        except Exception as e:
            print(f"    WARNING: Failed to load {subj_dir.name}: {e}")

        if n_loaded % 10 == 0:
            gc.collect()

    print(f"    Loaded saved band data for {n_loaded} subjects.")
    if n_loaded == 0:
        print("    Nothing loaded - check that band_data.npz / epochs_clean-epo.fif exist under "
              f"{recall_dir}")
        return

    run_analysis_for_groups(cond_accum, recall_name, cond_rep_epochs, "condition")
    del cond_accum, cond_rep_epochs
    gc.collect()


def run_behavioral_analysis(recall_dir, recall_name):
    """
    Main orchestrator: load per-subject data, run all analyses for
    condition, stress, and combined dimensions.
    """
    print(f"\n{'=' * 60}")
    print(f"BEHAVIORAL-EEG ANALYSIS - {recall_name}")
    print("=" * 60)

    subject_bands = _load_subject_band_data(recall_dir)
    if not subject_bands:
        print("    No subject data found. Skipping behavioral analysis.")
        return

    behav_dir = OUTPUT_ROOT / "behavioral_analysis" / recall_name
    behav_dir.mkdir(parents=True, exist_ok=True)

    all_desc = []
    all_comp = []
    all_corr = []

    # Stress is now a continuous covariate (see H3/H3b), so behavioral-EEG
    # grouping is by condition only.
    for dimension in ["condition"]:
        print(f"\n--- {dimension.upper()} ---")

        # Descriptive stats
        desc_df = describe_accuracy_by_group(subject_bands, dimension)
        all_desc.append(desc_df)
        print(f"    Descriptive stats:\n{desc_df.to_string(index=False)}")

        # Group comparisons
        comp_df = compare_accuracy_groups(subject_bands, dimension)
        all_comp.append(comp_df)
        if not comp_df.empty:
            print(f"    Group comparisons:\n{comp_df.to_string(index=False)}")

        # Accuracy-EEG correlations
        corr_df = compute_accuracy_eeg_correlations(subject_bands, dimension)
        all_corr.append(corr_df)
        sig = corr_df[corr_df["significant"]]
        if not sig.empty:
            print(f"    Significant correlations ({len(sig)} found):")
            print(f"    {sig[['band', 'time_window', 'channel', 'r_value', 'p_value']].to_string(index=False)}")
        else:
            print("    No significant correlations at p < 0.05.")

        # Plots
        plot_dir = behav_dir / dimension
        plot_accuracy_vs_eeg_scatter(subject_bands, dimension, plot_dir)
        plot_accuracy_eeg_heatmap(subject_bands, dimension, plot_dir)
        plot_accuracy_and_eeg_bars(subject_bands, dimension, plot_dir)

    # Save combined CSVs
    combined_desc = pd.concat(all_desc, ignore_index=True) if all_desc else None
    combined_comp = pd.concat(all_comp, ignore_index=True) if all_comp else None
    combined_corr = pd.concat(all_corr, ignore_index=True) if all_corr else None

    save_behavioral_results(combined_desc, combined_comp, combined_corr, behav_dir)
    print(f"\n  Behavioral-EEG analysis complete for {recall_name}")


# =============================================================================
# HYPOTHESIS TESTING - H1, H2, H3
# =============================================================================

def _extract_band_power(recall_dir, band="theta", key_channels=FRONTAL_CHANNELS):
    """
    Walk saved subject directories and extract mean power for a given band
    (default: theta) for each participant, averaged over `key_channels`
    (default: frontal-midline Fz/FCz - see FRONTAL_CHANNELS note above).

    Use band="beta" with the same frontal channels, or pass a different
    channel list, to test the alpha/beta predictions separately rather than
    reusing the whole-scalp KEY_CHANNELS average.

    Returns dict {subject_num: band_power_value}.
    """
    band_data = {}
    if not recall_dir.exists():
        print(f"    WARNING: {recall_dir} does not exist for {band} extraction.")
        return band_data

    for subj_dir in sorted(recall_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        band_path = subj_dir / "band_data.npz"
        if not band_path.exists():
            continue
        subj_num = get_participant_num(subj_dir)
        if subj_num is None:
            continue
        try:
            bdata = np.load(band_path, allow_pickle=True)
            if band in bdata:
                band_arr = np.array(bdata[band])
                if band_arr.ndim == 3:
                    band_arr = band_arr.mean(axis=0)  # avg over epochs
                # Average over the specified channels if available
                meta_path = subj_dir / "cwt_metadata.npz"
                if meta_path.exists():
                    meta = np.load(meta_path, allow_pickle=True)
                    ch_names = list(meta["ch_names"])
                    key_idx = [i for i, ch in enumerate(ch_names) if ch in key_channels]
                    if key_idx:
                        band_mean = band_arr[key_idx, :].mean()
                    else:
                        print(f"    WARNING: none of {key_channels} found for {subj_num}; "
                              f"falling back to whole-scalp average for this subject only.")
                        band_mean = band_arr.mean()
                else:
                    band_mean = band_arr.mean()
                band_data[subj_num] = float(band_mean)
        except Exception as e:
            print(f"    WARNING: Failed to extract {band} for {subj_num}: {e}")

    print(f"    Extracted {band} power ({key_channels}) for {len(band_data)} subjects from {recall_dir}")
    return band_data


def _extract_theta_power(recall_dir, key_channels=FRONTAL_CHANNELS):
    """Backwards-compatible wrapper: frontal-midline theta power per participant."""
    return _extract_band_power(recall_dir, band="theta", key_channels=key_channels)


def describe_band_power_by_condition(recall_dir, recall_name, participant_metadata, output_dir):
    """
    Descriptive (non-hypothesis) check: does theta/alpha/beta power at the
    frontal-midline electrodes (Fz, FCz) differ across the three conditions
    (control, ENG-SWA, SWA-ENG), separately for recall1 and recall2?

    Reuses _extract_band_power() so the numbers are extracted exactly the
    same way as the H3 mediation pathway (same channels, same averaging),
    just grouped by condition instead of correlated with stress/recall.

    Saves one CSV per recall session with descriptive stats (n, mean, sd)
    per band per condition, plus a Kruskal-Wallis test across the three
    conditions for each band (non-parametric, consistent with the rest of
    the H3 analysis since band power is non-normally distributed).

    Returns a pandas DataFrame with the results (also saved to disk).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for band in BANDS:
        band_power = _extract_band_power(recall_dir, band=band, key_channels=FRONTAL_CHANNELS)

        # group values by condition using the participant metadata CSV
        cond_values = {c: [] for c in CONDITIONS}
        for subj_num, value in band_power.items():
            meta = participant_metadata.get(subj_num)
            if meta is None:
                continue
            condition = meta.get("condition")
            if condition in cond_values:
                cond_values[condition].append(value)

        # descriptive stats per condition
        for condition in CONDITIONS:
            vals = cond_values[condition]
            if len(vals) == 0:
                continue
            rows.append({
                "recall": recall_name,
                "band": band,
                "condition": condition,
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            })

        # Kruskal-Wallis across the three conditions for this band
        group_lists = [cond_values[c] for c in CONDITIONS if len(cond_values[c]) > 0]
        if len(group_lists) >= 2 and all(len(g) > 0 for g in group_lists):
            try:
                h_stat, p_val = scipy_stats.kruskal(*group_lists)
                rows.append({
                    "recall": recall_name,
                    "band": band,
                    "condition": "KRUSKAL_WALLIS_ACROSS_CONDITIONS",
                    "n": sum(len(g) for g in group_lists),
                    "mean": float(h_stat),   # H statistic stored here for convenience
                    "sd": float(p_val),      # p-value stored here for convenience
                })
                print(f"    [{recall_name}] {band}: Kruskal-Wallis H={h_stat:.3f}, p={p_val:.3f}")
            except Exception as e:
                print(f"    WARNING: Kruskal-Wallis failed for {band} ({recall_name}): {e}")

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"band_power_by_condition_{recall_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"    Saved -> {csv_path}")
    return df


def test_h1_retrieval_practice(recall_data, output_dir, recall_col="recall2"):
    """
    H1: Retrieval practice -> higher recall accuracy.

    `recall_col` selects which recall test to run on ("recall1" or "recall2").
    Recall2 (the delayed test) is the primary hypothesis test; recall1 is run
    as a comparison/control check (is there already a group difference before
    the delay/retrieval-practice manipulation had a chance to matter?).

    Comparisons:
      1. One-way ANOVA across all three conditions (control, ENG-SWA, SWA-ENG)
      2. Pairwise Welch t-tests with Bonferroni correction:
           a) Control vs ENG-SWA
           b) Control vs SWA-ENG
           c) Control vs (ENG-SWA + SWA-ENG combined)
      Descriptives: n, mean, SD, SEM for each group (raw scores and % out of 30).
    """
    acc_key = f"accuracy_{recall_col}"
    print(f"\n{'='*60}")
    print(f"HYPOTHESIS TEST - H1: Retrieval Practice Effect ({recall_col})")
    print("="*60)

    # -- Collect scores per condition ------------------------------------------
    groups = {"control": [], "ENG-SWA": [], "SWA-ENG": []}

    for subj_num, meta in PARTICIPANT_METADATA.items():
        acc = meta.get(acc_key)
        if acc is None:
            continue
        cond = meta["condition"]
        if cond in groups:
            groups[cond].append(float(acc))

    control  = np.array(groups["control"])
    eng_swa  = np.array(groups["ENG-SWA"])
    swa_eng  = np.array(groups["SWA-ENG"])
    combined = np.concatenate([eng_swa, swa_eng])   # both retrieval practice conditions

    MAX_SCORE = 30  # out of 30 words

    def descriptives(arr, label):
        n    = len(arr)
        mean = float(np.mean(arr)) if n > 0 else np.nan
        sd   = float(np.std(arr, ddof=1)) if n > 1 else np.nan
        sem  = float(scipy_stats.sem(arr)) if n > 1 else np.nan
        pct  = round(mean / MAX_SCORE * 100, 2) if not np.isnan(mean) else np.nan
        print(f"    {label:30s}  n={n:3d}  mean={mean:5.2f}  SD={sd:5.2f}  "
              f"SEM={sem:5.2f}  ({pct}% of {MAX_SCORE})")
        return {"group": label, "n": n, "mean": round(mean, 4), "sd": round(sd, 4),
                "sem": round(sem, 4), "mean_pct": pct}

    print("\n  -- Descriptive Statistics --")
    desc = [
        descriptives(control,  "Control"),
        descriptives(eng_swa,  "ENG-SWA"),
        descriptives(swa_eng,  "SWA-ENG"),
        descriptives(combined, "ENG-SWA + SWA-ENG (combined)"),
    ]

    # -- Normality -------------------------------------------------------------
    print("\n  -- Shapiro-Wilk Normality Tests --")
    normality = {}
    for label, arr in [("control", control), ("ENG-SWA", eng_swa),
                        ("SWA-ENG", swa_eng), ("combined", combined)]:
        if len(arr) >= 3:
            _, p = scipy_stats.shapiro(arr)
            normality[label] = round(float(p), 6)
            print(f"    {label:30s}  W p={p:.4f}  {'normal OK' if p > 0.05 else 'NOT normal X'}")
        else:
            normality[label] = None
            print(f"    {label:30s}  too few observations")

    # -- One-way ANOVA: control vs ENG-SWA vs SWA-ENG -------------------------
    print("\n  -- One-Way ANOVA (Control vs ENG-SWA vs SWA-ENG) --")
    all_three = [control, eng_swa, swa_eng]
    if all(len(g) >= 2 for g in all_three):
        f_stat, p_anova = scipy_stats.f_oneway(*all_three)
        all_vals   = np.concatenate(all_three)
        grand_mean = np.mean(all_vals)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in all_three)
        ss_total   = np.sum((all_vals - grand_mean)**2)
        eta2       = ss_between / ss_total if ss_total > 0 else np.nan
        print(f"    F = {f_stat:.3f},  p = {p_anova:.4f},  eta^2 = {eta2:.4f}")
        print(f"    {'Significant OK' if p_anova < 0.05 else 'Not significant X'}")
        anova_results = {"F": round(float(f_stat), 4), "p": round(float(p_anova), 6),
                         "eta2": round(float(eta2), 4), "significant": bool(p_anova < 0.05)}
    else:
        print("    Not enough data for ANOVA.")
        anova_results = {}

    # -- Pairwise Welch t-tests (Bonferroni corrected for 3 comparisons) -------
    print("\n  -- Pairwise Welch t-tests (Bonferroni corrected, k=3) --")
    comparisons = [
        ("Control vs ENG-SWA",            control, eng_swa),
        ("Control vs SWA-ENG",            control, swa_eng),
        ("Control vs Combined (RP)",       control, combined),
    ]
    N_COMPARISONS = 3
    pairwise_results = []

    for label, g1, g2 in comparisons:
        if len(g1) < 2 or len(g2) < 2:
            print(f"    {label}: insufficient data.")
            continue
        t_stat, p_raw = scipy_stats.ttest_ind(g2, g1, equal_var=False)
        p_bonf = min(float(p_raw) * N_COMPARISONS, 1.0)
        pooled_sd = np.sqrt((np.var(g1, ddof=1) + np.var(g2, ddof=1)) / 2)
        cohens_d  = (np.mean(g2) - np.mean(g1)) / pooled_sd if pooled_sd > 0 else np.nan
        diff_mean = np.mean(g2) - np.mean(g1)
        sig = p_bonf < 0.05

        print(f"    {label}")
        print(f"      t = {t_stat:.3f},  p (raw) = {p_raw:.4f},  "
              f"p (Bonf) = {p_bonf:.4f},  d = {cohens_d:.3f},  "
              f"Deltamean = {diff_mean:+.2f}  "
              f"{'OK' if sig else 'X'}")

        pairwise_results.append({
            "comparison": label,
            "t": round(float(t_stat), 4),
            "p_raw": round(float(p_raw), 6),
            "p_bonferroni": round(float(p_bonf), 6),
            "cohens_d": round(float(cohens_d), 4),
            "mean_diff": round(float(diff_mean), 4),
            "significant_bonf": sig,
        })

    # -- Save ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    full_results = {
        "hypothesis": "H1: Retrieval practice -> higher recall accuracy",
        "recall": recall_col,
        "max_score": MAX_SCORE,
        "descriptives": desc,
        "normality_shapiro_p": normality,
        "anova": anova_results,
        "pairwise_welch_bonferroni": pairwise_results,
    }

    with open(output_dir / f"h1_retrieval_practice_{recall_col}.json", "w") as f:
        json.dump(full_results, f, indent=2)

    # Also save a clean CSV of descriptives
    pd.DataFrame(desc).to_csv(output_dir / f"h1_descriptives_{recall_col}.csv", index=False)
    pd.DataFrame(pairwise_results).to_csv(output_dir / f"h1_pairwise_{recall_col}.csv", index=False)

    print(f"\n    Saved -> {output_dir / f'h1_retrieval_practice_{recall_col}.json'}")
    print(f"    Saved -> {output_dir / f'h1_descriptives_{recall_col}.csv'}")
    print(f"    Saved -> {output_dir / f'h1_pairwise_{recall_col}.csv'}")

    # Overall H1 support: any pairwise comparison significant after correction?
    any_sig = any(r["significant_bonf"] for r in pairwise_results)
    full_results["h1_supported"] = any_sig
    return full_results


def test_h2_language_order_interaction(recall_data, output_dir, recall_col="recall2"):
    """
    H2: ENG-SWA shows stronger retrieval practice effect than SWA-ENG.
    2x2 ANOVA: condition_type (control vs experimental) x language_order (ENG-SWA vs SWA-ENG).
    Since control has no language order, we use a one-way ANOVA with 3 groups
    and planned contrasts to test the interaction.

    `recall_col` selects which recall test to run on ("recall1" or "recall2").
    Recall2 is the primary hypothesis test; recall1 is a comparison/control check.
    """
    acc_key = f"accuracy_{recall_col}"
    print(f"\n{'='*60}")
    print(f"HYPOTHESIS TEST - H2: Language Order Interaction ({recall_col})")
    print("="*60)

    # Group data
    control_accs = []
    exp_eng_swa_accs = []
    exp_swa_eng_accs = []

    for subj_num, meta in PARTICIPANT_METADATA.items():
        cond  = meta["condition"]
        acc = meta.get(acc_key)
        if acc is None:
            continue
        if cond == "control":
            control_accs.append(float(acc))
        elif cond == "ENG-SWA":
            exp_eng_swa_accs.append(float(acc))
        elif cond == "SWA-ENG":
            exp_swa_eng_accs.append(float(acc))

    if len(control_accs) < 2 or len(exp_eng_swa_accs) < 2 or len(exp_swa_eng_accs) < 2:
        print("    WARNING: Insufficient data for H2 (need >= 2 per group).")
        return None

    results = {
        "hypothesis": "H2: ENG-SWA shows stronger retrieval practice effect than SWA-ENG",
        "recall": recall_col,
        "control_n": len(control_accs),
        "exp_eng_swa_n": len(exp_eng_swa_accs),
        "exp_swa_eng_n": len(exp_swa_eng_accs),
        "control_mean": round(float(np.mean(control_accs)), 4),
        "exp_eng_swa_mean": round(float(np.mean(exp_eng_swa_accs)), 4),
        "exp_swa_eng_mean": round(float(np.mean(exp_swa_eng_accs)), 4),
    }

    # Shapiro-Wilk for each group
    for name, accs in [("control", control_accs), ("eng_swa", exp_eng_swa_accs), ("swa_eng", exp_swa_eng_accs)]:
        _, p = scipy_stats.shapiro(accs)
        results[f"{name}_normal_p"] = round(float(p), 6)

    # One-way ANOVA across 3 groups
    f_stat, p_anova = scipy_stats.f_oneway(control_accs, exp_eng_swa_accs, exp_swa_eng_accs)
    results["anova_f"] = round(float(f_stat), 4)
    results["anova_p"] = round(float(p_anova), 6)
    results["anova_significant"] = bool(p_anova < 0.05)

    # Eta-squared (effect size for ANOVA)
    all_vals = np.concatenate([control_accs, exp_eng_swa_accs, exp_swa_eng_accs])
    grand_mean = np.mean(all_vals)
    ss_between = (sum((np.mean(g) - grand_mean) ** 2 * len(g) for g in [control_accs, exp_eng_swa_accs, exp_swa_eng_accs]))
    ss_total = sum((x - grand_mean) ** 2 for x in all_vals)
    results["eta_squared"] = round(float(ss_between / ss_total), 6) if ss_total > 0 else 0

    # Post-hoc pairwise comparisons with Bonferroni correction
    pairwise = []
    comparisons = [
        ("Control vs ENG-SWA", control_accs, exp_eng_swa_accs),
        ("Control vs SWA-ENG", control_accs, exp_swa_eng_accs),
        ("ENG-SWA vs SWA-ENG", exp_eng_swa_accs, exp_swa_eng_accs),
    ]
    n_tests = len(comparisons)
    for label, g1, g2 in comparisons:
        _, p = scipy_stats.ttest_ind(g1, g2, equal_var=False)
        p_corr = min(p * n_tests, 1.0)
        d = (np.mean(g2) - np.mean(g1)) / np.sqrt((np.var(g1, ddof=1) + np.var(g2, ddof=1)) / 2)
        pairwise.append({
            "comparison": label,
            "p_raw": round(float(p), 6),
            "p_bonferroni": round(float(p_corr), 6),
            "cohens_d": round(float(d), 4),
            "significant": bool(p_corr < 0.05),
        })
    results["post_hoc"] = pairwise

    # Interaction test: is the retrieval practice effect (exp - control)
    # larger for ENG-SWA than for SWA-ENG?
    rp_eng_swa = np.mean(exp_eng_swa_accs) - np.mean(control_accs)
    rp_swa_eng = np.mean(exp_swa_eng_accs) - np.mean(control_accs)
    results["rp_effect_eng_swa"] = round(float(rp_eng_swa), 4)
    results["rp_effect_swa_eng"] = round(float(rp_swa_eng), 4)
    results["interaction_direction"] = "ENG-SWA stronger" if rp_eng_swa > rp_swa_eng else "SWA-ENG stronger"

    # Simple t-test on the interaction contrast
    # Compare (ENG-SWA - Control) vs (SWA-ENG - Control) using bootstrap
    n_boot = 10000
    boot_diffs = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx1 = rng.choice(len(exp_eng_swa_accs), len(exp_eng_swa_accs), replace=True)
        idx2 = rng.choice(len(exp_swa_eng_accs), len(exp_swa_eng_accs), replace=True)
        idx3 = rng.choice(len(control_accs), len(control_accs), replace=True)
        diff1 = np.mean(np.array(exp_eng_swa_accs)[idx1]) - np.mean(np.array(control_accs)[idx3])
        diff2 = np.mean(np.array(exp_swa_eng_accs)[idx2]) - np.mean(np.array(control_accs)[idx3])
        boot_diffs.append(diff1 - diff2)
    boot_diffs = np.array(boot_diffs)
    p_interaction = 2 * min(np.mean(boot_diffs >= 0), np.mean(boot_diffs <= 0))
    results["bootstrap_p_interaction"] = round(float(p_interaction), 6)
    results["bootstrap_ci_lower"] = round(float(np.percentile(boot_diffs, 2.5)), 4)
    results["bootstrap_ci_upper"] = round(float(np.percentile(boot_diffs, 97.5)), 4)
    results["interaction_significant"] = bool(p_interaction < 0.05)

    print(f"    Control: n={results['control_n']}, mean={results['control_mean']}")
    print(f"    ENG-SWA: n={results['exp_eng_swa_n']}, mean={results['exp_eng_swa_mean']}")
    print(f"    SWA-ENG: n={results['exp_swa_eng_n']}, mean={results['exp_swa_eng_mean']}")
    print(f"    ANOVA: F={results['anova_f']}, p={results['anova_p']}, eta^2={results['eta_squared']}")
    print(f"    RP effect ENG-SWA: {results['rp_effect_eng_swa']}")
    print(f"    RP effect SWA-ENG: {results['rp_effect_swa_eng']}")
    print(f"    Interaction (bootstrap): p={results['bootstrap_p_interaction']}")
    for ph in pairwise:
        print(f"    {ph['comparison']}: p_bonf={ph['p_bonferroni']}, d={ph['cohens_d']}")
    print(f"    Significant: ANOVA={results['anova_significant']}, Interaction={results['interaction_significant']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"h2_language_order_{recall_col}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Saved -> {output_dir / f'h2_language_order_{recall_col}.json'}")

    return results


def test_recall_change_score(output_dir):
    """
    Recall1 -> Recall2 change-score analysis, per condition.

    This is the direct test of whether retrieval practice caused a bigger
    GAIN over time, rather than just a higher endpoint at recall2:
      1. For each participant with both Recall1 and Recall2, compute
         change = recall2 - recall1.
      2. Paired test (Wilcoxon signed-rank, since n per group is small and
         change scores are often non-normal) per condition: did that group's
         recall change significantly from recall1 to recall2?
      3. One-way ANOVA (+ Kruskal-Wallis as a robust check) on the change
         scores across the three conditions: does the amount of
         change differ by condition/language order?

    Participants missing either Recall1 or Recall2 are skipped automatically.
    """
    print(f"\n{'='*60}")
    print("RECALL1 -> RECALL2 CHANGE SCORE ANALYSIS")
    print("="*60)

    groups = {"control": [], "ENG-SWA": [], "SWA-ENG": []}
    n_skipped = 0
    for subj_num, meta in PARTICIPANT_METADATA.items():
        r1 = meta.get("accuracy_recall1")
        r2 = meta.get("accuracy_recall2")
        cond = meta["condition"]
        if r1 is None or r2 is None or cond not in groups:
            n_skipped += 1
            continue
        groups[cond].append({"subj": subj_num, "recall1": float(r1),
                              "recall2": float(r2), "change": float(r2) - float(r1)})

    print(f"    Skipped {n_skipped} participants missing Recall1 and/or Recall2.")

    results = {"analysis": "Recall1 -> Recall2 change score by condition"}
    per_condition = {}

    for cond, rows in groups.items():
        n = len(rows)
        if n == 0:
            print(f"    {cond}: no complete pairs - skipping.")
            continue

        r1_vals = np.array([r["recall1"] for r in rows])
        r2_vals = np.array([r["recall2"] for r in rows])
        change  = r2_vals - r1_vals

        entry = {
            "n": n,
            "mean_recall1": round(float(np.mean(r1_vals)), 4),
            "mean_recall2": round(float(np.mean(r2_vals)), 4),
            "mean_change": round(float(np.mean(change)), 4),
            "sd_change": round(float(np.std(change, ddof=1)), 4) if n > 1 else np.nan,
            "sem_change": round(float(scipy_stats.sem(change)), 4) if n > 1 else np.nan,
        }

        # Paired test: did this group change from recall1 to recall2?
        if n >= 2:
            try:
                w_stat, w_p = scipy_stats.wilcoxon(r2_vals, r1_vals)
                entry["wilcoxon_stat"] = round(float(w_stat), 4)
                entry["wilcoxon_p"] = round(float(w_p), 6)
                entry["wilcoxon_significant"] = bool(w_p < 0.05)
            except ValueError as e:
                # e.g. all differences are zero
                entry["wilcoxon_error"] = str(e)
            t_stat, t_p = scipy_stats.ttest_rel(r2_vals, r1_vals)
            entry["paired_t_stat"] = round(float(t_stat), 4)
            entry["paired_t_p"] = round(float(t_p), 6)
            entry["paired_t_significant"] = bool(t_p < 0.05)

        print(f"    {cond:10s}  n={n:3d}  mean_recall1={entry['mean_recall1']:6.2f}  "
              f"mean_recall2={entry['mean_recall2']:6.2f}  mean_change={entry['mean_change']:+6.2f}  "
              f"paired_t_p={entry.get('paired_t_p')}")

        per_condition[cond] = entry

    results["per_condition"] = per_condition

    # Does the amount of change differ by condition?
    change_groups = {cond: [r["change"] for r in rows] for cond, rows in groups.items() if rows}
    if len(change_groups) >= 3 and all(len(v) >= 2 for v in change_groups.values()):
        vals = list(change_groups.values())
        f_stat, p_anova = scipy_stats.f_oneway(*vals)
        kw_stat, kw_p = scipy_stats.kruskal(*vals)
        results["change_score_anova"] = {
            "F": round(float(f_stat), 4), "p": round(float(p_anova), 6),
            "significant": bool(p_anova < 0.05),
        }
        results["change_score_kruskal"] = {
            "H": round(float(kw_stat), 4), "p": round(float(kw_p), 6),
            "significant": bool(kw_p < 0.05),
        }
        print(f"\n    Change score ANOVA across conditions: F={results['change_score_anova']['F']}, "
              f"p={results['change_score_anova']['p']}")
        print(f"    Change score Kruskal-Wallis (robust check): H={results['change_score_kruskal']['H']}, "
              f"p={results['change_score_kruskal']['p']}")
    else:
        print("    Not enough groups/data for a change-score group comparison.")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "recall_change_score.json", "w") as f:
        json.dump(results, f, indent=2)

    rows_out = []
    for cond, rows in groups.items():
        for r in rows:
            rows_out.append({"condition": cond, **r})
    if rows_out:
        pd.DataFrame(rows_out).to_csv(output_dir / "recall_change_score_by_subject.csv", index=False)

    print(f"    Saved -> {output_dir / 'recall_change_score.json'}")
    print(f"    Saved -> {output_dir / 'recall_change_score_by_subject.csv'}")

    return results


def _shapiro_report(label, arr):
    """
    Run + print a Shapiro-Wilk normality check on a 1D array. Mirrors the
    normality block already used in test_h1_retrieval_practice, applied
    here to the H3/H3b/H3c continuous variables (stress, band power,
    recall), which previously had no normality check at all.
    Returns the p-value (rounded) or None if there are too few observations.
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 3:
        print(f"    {label:20s}  too few observations for Shapiro-Wilk")
        return None
    _, p = scipy_stats.shapiro(arr)
    print(f"    {label:20s}  W p={p:.4f}  {'normal OK' if p > 0.05 else 'NOT normal X'}")
    return round(float(p), 6)


def _sensitivity_power(n, alpha=0.05, power=0.80):
    """
    A-priori-style sensitivity check for a bivariate correlation test at
    sample size n:
      (1) the minimum |r| that would even reach p < alpha, and
      (2) the minimum |r| this design has `power` (default 80%) chance of
          detecting at all, via the standard Fisher-z approximation
          (Cohen, 1988).
    For the *mediated* (indirect) effect specifically, published simulation
    tables (Fritz & MacKinnon, 2007, Psychological Science, 18(3), 233-239)
    indicate bias-corrected bootstrap mediation tests typically need
    roughly N=71-148 for 80% power, depending on the size of the a- and
    b-paths - so this single-correlation number is a *lower bound* on what
    the mediation test itself needs, not the mediation power directly.
    """
    df = n - 2
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, df)
    r_crit = t_crit / np.sqrt(t_crit ** 2 + df)
    z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
    z_pow = scipy_stats.norm.ppf(power)
    r80 = np.tanh((z_crit + z_pow) / np.sqrt(n - 3))
    return {
        "n": n,
        "r_significant_at_p05": round(float(r_crit), 4),
        "r_needed_for_80pct_power": round(float(r80), 4),
        "mediation_power_note": ("Fritz & MacKinnon (2007) report ~71-148 participants "
                                  "needed for 80% power in bias-corrected bootstrap mediation, "
                                  "depending on a/b path sizes; at this N, treat mediation power "
                                  "as at or below the single-correlation figures above."),
    }


def _standardized_paths(stress_vals, med_vals, recall_vals):
    """
    Standardized (z-scored) version of the a / b / c' / c paths, on top of
    the raw-unit slopes already computed elsewhere in each H3* function.
    Raw slopes depend on the arbitrary scales of stress/band-power/recall,
    which made path_c_total_effect look like it "disagreed" with the
    Pearson stress-recall r reported a few lines above it - they were
    never the same quantity (slope vs. correlation). Standardizing first
    puts every path in SD units and makes a/c directly comparable to the
    r-values reported earlier (for a single predictor, standardized beta
    == Pearson r), so this block should read as the more citable one.
    """
    def _z(x):
        x = np.asarray(x, dtype=float)
        return (x - x.mean()) / x.std(ddof=1)

    n = len(stress_vals)
    sz, mz, rz = _z(stress_vals), _z(med_vals), _z(recall_vals)
    Xa_z = np.column_stack([np.ones(n), sz])
    a_z = np.linalg.lstsq(Xa_z, mz, rcond=None)[0][1]
    c_z = np.linalg.lstsq(Xa_z, rz, rcond=None)[0][1]
    Xbc_z = np.column_stack([np.ones(n), sz, mz])
    coef_z = np.linalg.lstsq(Xbc_z, rz, rcond=None)[0]
    return {
        "a_stress_mediator": round(float(a_z), 4),
        "b_mediator_recall_partial": round(float(coef_z[2]), 4),
        "c_prime_stress_recall_ctrl": round(float(coef_z[1]), 4),
        "c_total": round(float(c_z), 4),
        "indirect_ab": round(float(a_z * coef_z[2]), 4),
    }


def _condition_controlled_b_path(combined, stress_vals, med_vals, recall_vals):
    """
    Exploratory robustness check: re-estimate the b-path (mediator ->
    recall) controlling for BOTH stress and condition, instead of just
    stress. Condition drove large behavioral effects in H1/H2 but was
    intentionally left out of H3's pre-registered mediation model; this
    reports what happens if it's added back in, labeled explicitly as
    exploratory rather than silently changing the primary model.
    """
    try:
        cond_list = [PARTICIPANT_METADATA[c["subj"]]["condition"] for c in combined]
        cond_dummies = pd.get_dummies(cond_list, drop_first=True).values.astype(float)
        n = len(combined)
        X = np.column_stack([np.ones(n), stress_vals, med_vals, cond_dummies])
        coef = np.linalg.lstsq(X, recall_vals, rcond=None)[0]
        return round(float(coef[2]), 4)
    except Exception as e:
        return f"error: {e}"


def test_h3_stress_theta_recall(recall_dir, recall2_data, output_dir):
    """
    H3: Higher stress -> reduced theta power -> poorer recall.
    Three-part mediation:
      a) Stress correlation with theta power
      b) Theta power correlation with recall accuracy
      c) Stress correlation with recall accuracy
      d) Mediation: does theta power partially mediate stress -> recall?
    """
    print(f"\n{'='*60}")
    print("HYPOTHESIS TEST - H3: Stress -> Theta Power -> Recall")
    print("="*60)

    results = {"hypothesis": "H3: Higher stress -> reduced theta power -> poorer recall"}

    # 1. Extract theta power per participant
    theta_by_subj = _extract_theta_power(recall_dir)

    # 2. Build combined dataset: stress (raw numeric), theta, recall2
    combined = []
    for subj_num, meta in PARTICIPANT_METADATA.items():
        acc_r2 = meta.get("accuracy_recall2")
        stress_num = meta.get("stress")
        if acc_r2 is None or stress_num is None:
            continue
        theta = theta_by_subj.get(subj_num)
        if theta is None:
            continue
        combined.append({
            "subj":       subj_num,
            "stress_num": stress_num,
            "theta":      theta,
            "recall2":    float(acc_r2),
        })

    if len(combined) < 5:
        print(f"    WARNING: Only {len(combined)} subjects with complete data for H3.")
        return None

    stress_vals = np.array([c["stress_num"] for c in combined])
    theta_vals = np.array([c["theta"] for c in combined])
    recall_vals = np.array([c["recall2"] for c in combined])

    results["n_subjects"] = len(combined)

    # -- Normality (Shapiro-Wilk), added: previously H3/H3b/H3c had no
    # normality check at all, unlike H1. -------------------------------------
    print("\n    -- Shapiro-Wilk Normality Checks --")
    results["normality_shapiro_p"] = {
        "stress":  _shapiro_report("stress", stress_vals),
        "theta":   _shapiro_report("theta", theta_vals),
        "recall2": _shapiro_report("recall2", recall_vals),
    }

    # -- Sensitivity / power, added: quantifies what this N could realistically
    # detect, instead of only reporting "not significant". ------------------
    results["sensitivity"] = _sensitivity_power(results["n_subjects"])

    # a) Stress vs Theta correlation
    r_stress_theta, p_stress_theta = scipy_stats.pearsonr(stress_vals, theta_vals)
    results["stress_theta_r"] = round(float(r_stress_theta), 4)
    results["stress_theta_p"] = round(float(p_stress_theta), 6)
    results["stress_theta_significant"] = bool(p_stress_theta < 0.05)

    # b) Theta vs Recall correlation
    r_theta_recall, p_theta_recall = scipy_stats.pearsonr(theta_vals, recall_vals)
    results["theta_recall_r"] = round(float(r_theta_recall), 4)
    results["theta_recall_p"] = round(float(p_theta_recall), 6)
    results["theta_recall_significant"] = bool(p_theta_recall < 0.05)

    # c) Stress vs Recall correlation
    r_stress_recall, p_stress_recall = scipy_stats.pearsonr(stress_vals, recall_vals)
    results["stress_recall_r"] = round(float(r_stress_recall), 4)
    results["stress_recall_p"] = round(float(p_stress_recall), 6)
    results["stress_recall_significant"] = bool(p_stress_recall < 0.05)

    # d) Mediation analysis (Baron & Kenny approach)
    # Path a:  stress -> theta            (unadjusted slope)
    # Path c:  stress -> recall           (unadjusted slope, total effect)
    # Path b / c': recall ~ stress + theta (theta and stress simultaneously) ->
    #   coefficient on theta = b (theta -> recall, controlling for stress)
    #   coefficient on stress = c' (stress -> recall, controlling for theta)
    n = len(combined)
    try:
        Xa = np.column_stack([np.ones(n), stress_vals])
        a = np.linalg.lstsq(Xa, theta_vals, rcond=None)[0][1]
        c_total = np.linalg.lstsq(Xa, recall_vals, rcond=None)[0][1]

        Xbc = np.column_stack([np.ones(n), stress_vals, theta_vals])
        recall_on_both = np.linalg.lstsq(Xbc, recall_vals, rcond=None)[0]
        b = recall_on_both[2]        # theta -> recall, controlling for stress
        c_prime = recall_on_both[1]  # stress -> recall, controlling for theta

        # ---- Bootstrapped mediation (replaces Sobel test) ----
        # The Sobel test assumes a*b is normally distributed, which is rarely true
        # in small/medium samples (Hayes, 2009; Preacher & Hayes, 2004, 2008). The
        # modern standard - what PROCESS does under the hood - is a nonparametric
        # percentile bootstrap on the indirect effect: resample subjects with
        # replacement, recompute a*b each time, and take the 2.5th/97.5th
        # percentiles of that distribution as the 95% CI. No normality assumption,
        # and it directly answers "is zero excluded from the plausible indirect
        # effects" rather than relying on a z-test of a non-normal statistic.
        N_BOOT = 5000
        rng = np.random.default_rng(42)  # seeded for reproducibility
        boot_indirect = np.empty(N_BOOT)
        n_obs = len(combined)

        for i in range(N_BOOT):
            idx = rng.integers(0, n_obs, size=n_obs)  # resample subjects w/ replacement
            s_b, t_b, r_b = stress_vals[idx], theta_vals[idx], recall_vals[idx]
            Xb = np.column_stack([np.ones(n_obs), s_b, t_b])
            try:
                # a path: stress -> theta
                a_b = np.linalg.lstsq(np.column_stack([np.ones(n_obs), s_b]), t_b, rcond=None)[0][1]
                # b path: theta -> recall, controlling for stress
                b_b = np.linalg.lstsq(Xb, r_b, rcond=None)[0][2]
                boot_indirect[i] = a_b * b_b
            except np.linalg.LinAlgError:
                boot_indirect[i] = np.nan

        boot_indirect = boot_indirect[~np.isnan(boot_indirect)]
        ci_low, ci_high = np.percentile(boot_indirect, [2.5, 97.5])
        boot_significant = bool(ci_low > 0 or ci_high < 0)  # CI excludes zero

        results["mediation"] = {
            "method": "bootstrap_percentile_CI",
            "n_boot": N_BOOT,
            "path_a_stress_theta": round(float(a), 4),
            "path_b_theta_recall_partial": round(float(b), 4),
            "path_c_prime_stress_recall_ctrl": round(float(c_prime), 4),
            "path_c_total_effect": round(float(c_total), 4),
            "indirect_effect": round(float(a * b), 4),
            "boot_ci_95_low": round(float(ci_low), 4),
            "boot_ci_95_high": round(float(ci_high), 4),
            "boot_significant": boot_significant,
            "mediation_type": "full" if abs(c_prime) < 0.1 and boot_significant else "partial" if boot_significant else "none",
        }
    except Exception as e:
        results["mediation_error"] = str(e)

    # -- Standardized paths, added: resolves path_c_total_effect (raw slope)
    # visually "disagreeing" with stress_recall_r (correlation) above - they
    # were never the same quantity. This block IS directly comparable to the
    # r-values above. ---------------------------------------------------------
    results["standardized_paths"] = _standardized_paths(stress_vals, theta_vals, recall_vals)

    # -- Condition-controlled b-path, added: exploratory robustness check,
    # NOT part of the pre-registered model (see docstring on the helper). ----
    results["b_path_theta_recall_ctrl_stress_and_condition_exploratory"] = \
        _condition_controlled_b_path(combined, stress_vals, theta_vals, recall_vals)

    # Summary
    print(f"    Subjects with complete data: {results['n_subjects']}")
    print(f"    a) Stress -> Theta: r={results['stress_theta_r']}, p={results['stress_theta_p']}")
    print(f"    b) Theta -> Recall: r={results['theta_recall_r']}, p={results['theta_recall_p']}")
    print(f"    c) Stress -> Recall: r={results['stress_recall_r']}, p={results['stress_recall_p']}")
    if "mediation" in results:
        m = results["mediation"]
        print(f"    Indirect effect (a*b): {m['indirect_effect']}")
        print(f"    Bootstrap 95% CI: [{m['boot_ci_95_low']}, {m['boot_ci_95_high']}]  "
              f"({'excludes zero' if m['boot_significant'] else 'includes zero'})")
        print(f"    Mediation type: {m['mediation_type']}")
    h3_supported = (results["stress_theta_significant"] and
                    results["theta_recall_significant"] and
                    results["stress_recall_significant"])
    results["h3_supported"] = h3_supported
    print(f"    H3 supported: {h3_supported}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "h3_stress_theta_recall.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Saved -> {output_dir / 'h3_stress_theta_recall.json'}")

    return results


def test_h3b_stress_beta_recall(recall_dir, recall2_data, output_dir):
    """
    Optional companion to H3, testing frontal beta power instead of theta.

    Per the stress literature, beta and theta are predicted to move in
    OPPOSITE directions with stress: alpha/theta tend to decrease under
    stress while beta tends to increase (associated with heightened
    arousal/anxiety and information-processing demand). So unlike H3
    (stress -> LOWER theta -> poorer recall), the beta version predicts
    stress -> HIGHER beta -> poorer recall. This function does not assume
    a sign - it just reports the same three correlations and bootstrapped
    mediation CI as test_h3_stress_theta_recall, using "beta" extracted
    over the same frontal channels, so you can check whether the sign
    actually comes out the way the literature predicts before reporting it
    as supporting/disconfirming H3's specificity claim.
    """
    print(f"\n{'='*60}")
    print("HYPOTHESIS TEST - H3b (exploratory): Stress -> Frontal Beta -> Recall")
    print("="*60)

    results = {"hypothesis": "H3b (exploratory): stress -> frontal beta -> recall; "
                              "predicted direction is opposite sign to H3's theta path"}

    beta_by_subj = _extract_band_power(recall_dir, band="beta", key_channels=FRONTAL_CHANNELS)

    combined = []
    for subj_num, meta in PARTICIPANT_METADATA.items():
        acc_r2 = meta.get("accuracy_recall2")
        stress_num = meta.get("stress")
        if acc_r2 is None or stress_num is None:
            continue
        beta = beta_by_subj.get(subj_num)
        if beta is None:
            continue
        combined.append({"subj": subj_num, "stress_num": stress_num,
                          "beta": beta, "recall2": float(acc_r2)})

    if len(combined) < 5:
        print(f"    WARNING: Only {len(combined)} subjects with complete data for H3b.")
        return None

    stress_vals = np.array([c["stress_num"] for c in combined])
    beta_vals = np.array([c["beta"] for c in combined])
    recall_vals = np.array([c["recall2"] for c in combined])
    n_obs = len(combined)
    results["n_subjects"] = n_obs

    print("\n    -- Shapiro-Wilk Normality Checks --")
    results["normality_shapiro_p"] = {
        "stress":  _shapiro_report("stress", stress_vals),
        "beta":    _shapiro_report("beta", beta_vals),
        "recall2": _shapiro_report("recall2", recall_vals),
    }
    results["sensitivity"] = _sensitivity_power(n_obs)

    r_stress_beta, p_stress_beta = scipy_stats.pearsonr(stress_vals, beta_vals)
    results["stress_beta_r"] = round(float(r_stress_beta), 4)
    results["stress_beta_p"] = round(float(p_stress_beta), 6)
    results["stress_beta_significant"] = bool(p_stress_beta < 0.05)
    results["direction_matches_literature"] = bool(r_stress_beta > 0)  # expect positive

    r_beta_recall, p_beta_recall = scipy_stats.pearsonr(beta_vals, recall_vals)
    results["beta_recall_r"] = round(float(r_beta_recall), 4)
    results["beta_recall_p"] = round(float(p_beta_recall), 6)

    r_stress_recall, p_stress_recall = scipy_stats.pearsonr(stress_vals, recall_vals)
    results["stress_recall_r"] = round(float(r_stress_recall), 4)
    results["stress_recall_p"] = round(float(p_stress_recall), 6)

    # Bootstrapped indirect effect (same approach as H3)
    N_BOOT = 5000
    rng = np.random.default_rng(42)
    boot_indirect = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n_obs, size=n_obs)
        s_b, t_b, r_b = stress_vals[idx], beta_vals[idx], recall_vals[idx]
        try:
            a_b = np.linalg.lstsq(np.column_stack([np.ones(n_obs), s_b]), t_b, rcond=None)[0][1]
            Xb = np.column_stack([np.ones(n_obs), s_b, t_b])
            b_b = np.linalg.lstsq(Xb, r_b, rcond=None)[0][2]
            boot_indirect[i] = a_b * b_b
        except np.linalg.LinAlgError:
            boot_indirect[i] = np.nan
    boot_indirect = boot_indirect[~np.isnan(boot_indirect)]
    ci_low, ci_high = np.percentile(boot_indirect, [2.5, 97.5])
    results["mediation"] = {
        "method": "bootstrap_percentile_CI",
        "n_boot": N_BOOT,
        "boot_ci_95_low": round(float(ci_low), 4),
        "boot_ci_95_high": round(float(ci_high), 4),
        "boot_significant": bool(ci_low > 0 or ci_high < 0),
    }

    results["standardized_paths"] = _standardized_paths(stress_vals, beta_vals, recall_vals)
    results["b_path_beta_recall_ctrl_stress_and_condition_exploratory"] = \
        _condition_controlled_b_path(combined, stress_vals, beta_vals, recall_vals)

    print(f"    Stress -> Beta: r={results['stress_beta_r']}, p={results['stress_beta_p']} "
          f"(literature predicts positive r: {'matches' if results['direction_matches_literature'] else 'does NOT match'})")
    print(f"    Beta -> Recall: r={results['beta_recall_r']}, p={results['beta_recall_p']}")
    print(f"    Bootstrap 95% CI on indirect effect: [{results['mediation']['boot_ci_95_low']}, "
          f"{results['mediation']['boot_ci_95_high']}]")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "h3b_stress_beta_recall.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Saved -> {output_dir / 'h3b_stress_beta_recall.json'}")

    return results


def test_h3c_stress_alpha_recall(recall_dir, recall2_data, output_dir):
    """
    Second exploratory companion to H3, testing frontal alpha power instead
    of theta (parallel to test_h3b_stress_beta_recall for beta).

    Alpha, like theta, is generally reported to decrease under stress and
    to relate to attentional/inhibitory control during memory tasks, so the
    predicted direction here is the SAME sign as H3's theta path (stress ->
    LOWER alpha -> poorer recall). As with H3b, this function does not
    assume the sign is correct - it reports the same three correlations,
    point-estimate mediation paths (a, b, c', c), and bootstrapped
    indirect-effect CI as test_h3_stress_theta_recall, using "alpha"
    extracted over the same frontal channels (Fz, FCz).
    """
    print(f"\n{'='*60}")
    print("HYPOTHESIS TEST - H3c (exploratory): Stress -> Frontal Alpha -> Recall")
    print("="*60)

    results = {"hypothesis": "H3c (exploratory): stress -> frontal alpha -> recall; "
                              "predicted direction is same sign as H3's theta path"}

    alpha_by_subj = _extract_band_power(recall_dir, band="alpha", key_channels=FRONTAL_CHANNELS)

    combined = []
    for subj_num, meta in PARTICIPANT_METADATA.items():
        acc_r2 = meta.get("accuracy_recall2")
        stress_num = meta.get("stress")
        if acc_r2 is None or stress_num is None:
            continue
        alpha = alpha_by_subj.get(subj_num)
        if alpha is None:
            continue
        combined.append({"subj": subj_num, "stress_num": stress_num,
                          "alpha": alpha, "recall2": float(acc_r2)})

    if len(combined) < 5:
        print(f"    WARNING: Only {len(combined)} subjects with complete data for H3c.")
        return None

    stress_vals = np.array([c["stress_num"] for c in combined])
    alpha_vals = np.array([c["alpha"] for c in combined])
    recall_vals = np.array([c["recall2"] for c in combined])
    n_obs = len(combined)
    results["n_subjects"] = n_obs

    print("\n    -- Shapiro-Wilk Normality Checks --")
    results["normality_shapiro_p"] = {
        "stress":  _shapiro_report("stress", stress_vals),
        "alpha":   _shapiro_report("alpha", alpha_vals),
        "recall2": _shapiro_report("recall2", recall_vals),
    }
    results["sensitivity"] = _sensitivity_power(n_obs)

    r_stress_alpha, p_stress_alpha = scipy_stats.pearsonr(stress_vals, alpha_vals)
    results["stress_alpha_r"] = round(float(r_stress_alpha), 4)
    results["stress_alpha_p"] = round(float(p_stress_alpha), 6)
    results["stress_alpha_significant"] = bool(p_stress_alpha < 0.05)
    results["direction_matches_literature"] = bool(r_stress_alpha < 0)  # expect negative, like theta

    r_alpha_recall, p_alpha_recall = scipy_stats.pearsonr(alpha_vals, recall_vals)
    results["alpha_recall_r"] = round(float(r_alpha_recall), 4)
    results["alpha_recall_p"] = round(float(p_alpha_recall), 6)

    r_stress_recall, p_stress_recall = scipy_stats.pearsonr(stress_vals, recall_vals)
    results["stress_recall_r"] = round(float(r_stress_recall), 4)
    results["stress_recall_p"] = round(float(p_stress_recall), 6)

    # Point-estimate mediation paths (same approach as H3)
    try:
        Xa = np.column_stack([np.ones(n_obs), stress_vals])
        a = np.linalg.lstsq(Xa, alpha_vals, rcond=None)[0][1]
        c_total = np.linalg.lstsq(Xa, recall_vals, rcond=None)[0][1]

        Xbc = np.column_stack([np.ones(n_obs), stress_vals, alpha_vals])
        recall_on_both = np.linalg.lstsq(Xbc, recall_vals, rcond=None)[0]
        b = recall_on_both[2]        # alpha -> recall, controlling for stress
        c_prime = recall_on_both[1]  # stress -> recall, controlling for alpha

        # Bootstrapped indirect effect (same approach as H3)
        N_BOOT = 5000
        rng = np.random.default_rng(42)
        boot_indirect = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, n_obs, size=n_obs)
            s_b, t_b, r_b = stress_vals[idx], alpha_vals[idx], recall_vals[idx]
            Xb = np.column_stack([np.ones(n_obs), s_b, t_b])
            try:
                a_b = np.linalg.lstsq(np.column_stack([np.ones(n_obs), s_b]), t_b, rcond=None)[0][1]
                b_b = np.linalg.lstsq(Xb, r_b, rcond=None)[0][2]
                boot_indirect[i] = a_b * b_b
            except np.linalg.LinAlgError:
                boot_indirect[i] = np.nan
        boot_indirect = boot_indirect[~np.isnan(boot_indirect)]
        ci_low, ci_high = np.percentile(boot_indirect, [2.5, 97.5])
        boot_significant = bool(ci_low > 0 or ci_high < 0)

        results["mediation"] = {
            "method": "bootstrap_percentile_CI",
            "n_boot": N_BOOT,
            "path_a_stress_alpha": round(float(a), 4),
            "path_b_alpha_recall_partial": round(float(b), 4),
            "path_c_prime_stress_recall_ctrl": round(float(c_prime), 4),
            "path_c_total_effect": round(float(c_total), 4),
            "indirect_effect": round(float(a * b), 4),
            "boot_ci_95_low": round(float(ci_low), 4),
            "boot_ci_95_high": round(float(ci_high), 4),
            "boot_significant": boot_significant,
            "mediation_type": "full" if abs(c_prime) < 0.1 and boot_significant else "partial" if boot_significant else "none",
        }
    except Exception as e:
        results["mediation_error"] = str(e)

    results["standardized_paths"] = _standardized_paths(stress_vals, alpha_vals, recall_vals)
    results["b_path_alpha_recall_ctrl_stress_and_condition_exploratory"] = \
        _condition_controlled_b_path(combined, stress_vals, alpha_vals, recall_vals)

    print(f"    Stress -> Alpha: r={results['stress_alpha_r']}, p={results['stress_alpha_p']} "
          f"(literature predicts negative r: {'matches' if results['direction_matches_literature'] else 'does NOT match'})")
    print(f"    Alpha -> Recall: r={results['alpha_recall_r']}, p={results['alpha_recall_p']}")
    if "mediation" in results:
        print(f"    Bootstrap 95% CI on indirect effect: [{results['mediation']['boot_ci_95_low']}, "
              f"{results['mediation']['boot_ci_95_high']}]")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "h3c_stress_alpha_recall.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"    Saved -> {output_dir / 'h3c_stress_alpha_recall.json'}")

    return results


def _condition_colors():
    """Consistent colorful palette used across all H1/H2 figures."""
    return {"control": "#4C72B0", "ENG-SWA": "#C44E52", "SWA-ENG": "#DD8452"}


def plot_recall_boxplots_by_condition(output_dir):
    """
    Colorful boxplots of recall accuracy by condition (H1) - saved as TWO
    separate figures, one for immediate recall (recall1) and one for delayed
    recall (recall2). Same visual style as the Polysemous/Monosemic accuracy
    boxplot (Figure 6): patch_artist boxes, distinct fill colors per group,
    black whiskers/medians, jittered raw points on top.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = _condition_colors()
    order = ["control", "ENG-SWA", "SWA-ENG"]
    nice_labels = {"control": "Control", "ENG-SWA": "ENG-SWA", "SWA-ENG": "SWA-ENG"}

    saved_paths = []

    for acc_key, title, fname in [
        ("accuracy_recall1", "Immediate Recall Accuracy by Condition", "h1_boxplot_immediate_recall.png"),
        ("accuracy_recall2", "Delayed Recall Accuracy by Condition", "h1_boxplot_delayed_recall.png"),
    ]:
        data, labels, box_colors = [], [], []
        for cond in order:
            vals = [meta[acc_key] for meta in PARTICIPANT_METADATA.values()
                    if meta["condition"] == cond and meta.get(acc_key) is not None]
            if not vals:
                continue
            data.append(vals)
            labels.append(nice_labels[cond])
            box_colors.append(colors[cond])

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, tick_labels=labels,
                         medianprops=dict(color="black", linewidth=1.8),
                         whiskerprops=dict(color="black"),
                         capprops=dict(color="black"),
                         flierprops=dict(marker="o", markerfacecolor="black",
                                          markersize=4, alpha=0.6))
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
            patch.set_edgecolor("black")

        # jittered raw points on top for transparency
        rng = np.random.RandomState(0)
        for i, vals in enumerate(data, start=1):
            jitter = rng.normal(0, 0.05, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color="black", alpha=0.35, s=16, zorder=3)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Word-pairs recalled")
        ax.set_xlabel("Condition")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        plot_path = output_dir / fname
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved boxplot -> {plot_path}")
        saved_paths.append(plot_path)

    return saved_paths


def plot_h2_boxplots(output_dir):
    """
    H2-focused boxplot: ENG-SWA vs SWA-ENG only, immediate and delayed recall
    side by side. Same colorful patch_artist styling as Figure 6.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = _condition_colors()
    order = ["ENG-SWA", "SWA-ENG"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5))
    for ax, acc_key, title in zip(
        axes,
        ["accuracy_recall1", "accuracy_recall2"],
        ["Immediate Recall", "Delayed Recall"],
    ):
        data, box_colors = [], []
        for cond in order:
            vals = [meta[acc_key] for meta in PARTICIPANT_METADATA.values()
                    if meta["condition"] == cond and meta.get(acc_key) is not None]
            data.append(vals)
            box_colors.append(colors[cond])

        bp = ax.boxplot(data, patch_artist=True, widths=0.5, labels=order,
                         medianprops=dict(color="black", linewidth=1.8),
                         whiskerprops=dict(color="black"),
                         capprops=dict(color="black"))
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
            patch.set_edgecolor("black")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Word-pairs recalled")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.suptitle("Desirable Difficulty: ENG-SWA vs SWA-ENG (H2)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = output_dir / "h2_recall_boxplots_eng_swa_vs_swa_eng.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved boxplot -> {plot_path}")
    return plot_path


def plot_h1_h2_significance_heatmap(h1_r1, h1_r2, h2_r1, h2_r2, output_dir):
    """
    p-value heatmap in the same style as the 'Significance of Language Use
    Context' figure: imshow + colorbar + annotated cell values, rows =
    comparisons, columns = Immediate/Delayed recall.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _pbonf(res, label):
        if not res:
            return np.nan
        for r in res.get("pairwise_welch_bonferroni", []):
            if r["comparison"] == label:
                return r["p_bonferroni"]
        return np.nan

    def _h2_pbonf(res, label):
        if not res:
            return np.nan
        for r in res.get("post_hoc", []):
            if r["comparison"] == label:
                return r["p_bonferroni"]
        return np.nan

    rows = [
        ("Control vs ENG-SWA", _pbonf(h1_r1, "Control vs ENG-SWA"), _pbonf(h1_r2, "Control vs ENG-SWA")),
        ("Control vs SWA-ENG", _pbonf(h1_r1, "Control vs SWA-ENG"), _pbonf(h1_r2, "Control vs SWA-ENG")),
        ("ENG-SWA vs SWA-ENG", _h2_pbonf(h2_r1, "ENG-SWA vs SWA-ENG"), _h2_pbonf(h2_r2, "ENG-SWA vs SWA-ENG")),
    ]

    labels = [r[0] for r in rows]
    matrix = np.array([[r[1], r[2]] for r in rows])

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, cmap="coolwarm_r", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Immediate Recall", "Delayed Recall"])
    ax.set_title("Bonferroni-corrected p-values: H1 & H2 pairwise comparisons")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            txt = "n/a" if np.isnan(val) else f"{val:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="white" if (not np.isnan(val) and val < 0.5) else "black")

    plt.colorbar(im, ax=ax, label="p (Bonferroni)")
    plt.tight_layout()
    plot_path = output_dir / "h1_h2_significance_heatmap.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved significance heatmap -> {plot_path}")
    return plot_path


def build_h1_h2_results_table(h1_r1, h1_r2, h2_r1, h2_r2, output_dir):
    """
    One consolidated, colour-coded results table (descriptives + inferential
    stats for H1 and H2, both recall trials) saved as CSV and as a rendered
    PNG table (green-shaded rows = significant after Bonferroni correction).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def _add_h1_rows(res, recall_label):
        if not res:
            return
        desc_by_group = {d["group"]: d for d in res["descriptives"]}
        anova = res.get("anova", {})
        for r in res.get("pairwise_welch_bonferroni", []):
            g2_label = r["comparison"].split(" vs ")[-1]
            d = desc_by_group.get(g2_label, {})
            rows.append({
                "Recall": recall_label,
                "Hypothesis": "H1",
                "Comparison": r["comparison"],
                "Mean (group)": d.get("mean"),
                "SD (group)": d.get("sd"),
                "n": d.get("n"),
                "ANOVA F": anova.get("F"),
                "ANOVA p": anova.get("p"),
                "eta2": anova.get("eta2"),
                "t": r["t"],
                "p_raw": r["p_raw"],
                "p_bonf": r["p_bonferroni"],
                "Cohen's d": r["cohens_d"],
                "Significant (Bonf)": r["significant_bonf"],
            })

    def _add_h2_rows(res, recall_label):
        if not res:
            return
        for r in res.get("post_hoc", []):
            rows.append({
                "Recall": recall_label,
                "Hypothesis": "H2",
                "Comparison": r["comparison"],
                "Mean (group)": None,
                "SD (group)": None,
                "n": None,
                "ANOVA F": res.get("anova_f"),
                "ANOVA p": res.get("anova_p"),
                "eta2": res.get("eta_squared"),
                "t": None,
                "p_raw": r["p_raw"],
                "p_bonf": r["p_bonferroni"],
                "Cohen's d": r["cohens_d"],
                "Significant (Bonf)": r["significant"],
            })

    _add_h1_rows(h1_r1, "Immediate")
    _add_h1_rows(h1_r2, "Delayed")
    _add_h2_rows(h2_r1, "Immediate")
    _add_h2_rows(h2_r2, "Delayed")

    df = pd.DataFrame(rows)
    csv_path = output_dir / "h1_h2_results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"    Saved results table (csv) -> {csv_path}")

    if df.empty:
        return df

    # ---- render as a coloured PNG table ----
    display_cols = ["Recall", "Hypothesis", "Comparison", "Mean (group)", "SD (group)",
                     "n", "ANOVA p", "t", "p_bonf", "Cohen's d", "Significant (Bonf)"]
    disp = df[display_cols].copy()
    for c in ["Mean (group)", "SD (group)", "ANOVA p", "t", "p_bonf", "Cohen's d"]:
        disp[c] = disp[c].apply(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    disp["n"] = disp["n"].apply(lambda v: "" if pd.isna(v) else f"{int(v)}")
    disp["Significant (Bonf)"] = disp["Significant (Bonf)"].map({True: "Yes", False: "No"})

    fig, ax = plt.subplots(figsize=(14, 0.55 * len(disp) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=disp.values, colLabels=disp.columns,
                    cellLoc="center", loc="center", colWidths=[0.09, 0.09, 0.19, 0.1, 0.09,
                                                                0.06, 0.09, 0.07, 0.08, 0.09, 0.13])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for j, col in enumerate(disp.columns):
        cell = tbl[0, j]
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", fontweight="bold")

    for i, sig in enumerate(df["Significant (Bonf)"], start=1):
        color = "#C6E5C6" if sig else "#FFFFFF"
        for j in range(len(disp.columns)):
            tbl[i, j].set_facecolor(color)

    plt.title("H1 & H2 Results Summary (green = significant after Bonferroni correction)",
               fontsize=11, fontweight="bold", pad=20)
    plt.tight_layout()
    png_path = output_dir / "h1_h2_results_table.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved results table (image) -> {png_path}")
    return df


def build_h3_results_table(h3, h3b, h3c, output_dir):
    """
    One consolidated H3/H3b/H3c table: stress-power-recall mediation results
    for theta (primary), beta and alpha (exploratory companions).

    Saved as CSV (full precision, all columns - for your appendix/data
    archive) and as a plain, thesis-style PNG table (APA-like: white
    background, horizontal rules only, no fill colours, r and p combined
    into one cell) that only keeps the columns a reader actually needs to
    follow the mediation logic: the three paths (a, b, c) and the indirect
    effect with its CI.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _row(res, band_label, power_key):
        if not res:
            return None
        med = res.get("mediation", {}) or {}
        return {
            "Band": band_label,
            "n": res.get("n_subjects"),
            "r (stress-power)": res.get(f"stress_{power_key}_r"),
            "p (stress-power)": res.get(f"stress_{power_key}_p"),
            "r (power-recall)": res.get(f"{power_key}_recall_r"),
            "p (power-recall)": res.get(f"{power_key}_recall_p"),
            "r (stress-recall)": res.get("stress_recall_r"),
            "p (stress-recall)": res.get("stress_recall_p"),
            "Indirect effect (a*b)": med.get("indirect_effect"),
            "Boot 95% CI low": med.get("boot_ci_95_low"),
            "Boot 95% CI high": med.get("boot_ci_95_high"),
            "Mediation significant": med.get("boot_significant", False),
        }

    rows = [
        _row(h3, "Theta (primary, frontal)", "theta"),
        _row(h3b, "Beta (exploratory, frontal)", "beta"),
        _row(h3c, "Alpha (exploratory, frontal)", "alpha"),
    ]
    rows = [r for r in rows if r is not None]

    df = pd.DataFrame(rows)
    csv_path = output_dir / "h3_results_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"    Saved H3 results table (csv, full) -> {csv_path}")

    if df.empty:
        return df

    # ---- Build the simplified display table -------------------------------
    def _fmt_r_p(r, p):
        if pd.isna(r):
            return "-"
        if pd.isna(p):
            return f"{r:.3f}"
        p_str = "<.001" if p < .001 else f"= {p:.3f}"
        p_str = p_str.replace("0.", ".")
        return f"{r:.3f} (p {p_str})"

    def _fmt_ci(lo, hi):
        if pd.isna(lo) or pd.isna(hi):
            return "-"
        return f"[{lo:.3f}, {hi:.3f}]"

    disp = pd.DataFrame({
        "Band": df["Band"].str.replace(" (primary, frontal)", "", regex=False)
                            .str.replace(" (exploratory, frontal)", "", regex=False),
        "n": df["n"],
        "a-path\n(Stress -> Power)": [
            _fmt_r_p(r, p) for r, p in zip(df["r (stress-power)"], df["p (stress-power)"])
        ],
        "b-path\n(Power -> Recall)": [
            _fmt_r_p(r, p) for r, p in zip(df["r (power-recall)"], df["p (power-recall)"])
        ],
        "c-path\n(Stress -> Recall)": [
            _fmt_r_p(r, p) for r, p in zip(df["r (stress-recall)"], df["p (stress-recall)"])
        ],
        "Indirect\neffect (ab)": df["Indirect effect (a*b)"].apply(
            lambda v: "-" if pd.isna(v) else f"{v:.3f}"),
        "95% CI": [
            _fmt_ci(lo, hi) for lo, hi in zip(df["Boot 95% CI low"], df["Boot 95% CI high"])
        ],
        "Mediation": df["Mediation significant"].map({True: "Significant", False: "n.s."}),
    })

    n_rows, n_cols = len(disp), len(disp.columns)
    fig, ax = plt.subplots(figsize=(12, 0.9 * n_rows + 1.6))
    ax.axis("off")

    col_widths = [0.15, 0.05, 0.19, 0.19, 0.19, 0.11, 0.16, 0.12]
    tbl = ax.table(cellText=disp.values, colLabels=disp.columns, cellLoc="center",
                    loc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    # Plain white table, APA-style: bold header + top/bottom rules only,
    # a rule under the header row, no fill colour anywhere.
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("white")
        cell.set_text_props(ha="center")
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.visible_edges = "B"
            cell.set_linewidth(1.4)
        elif row == n_rows:
            cell.visible_edges = "B"
            cell.set_linewidth(1.2)
        else:
            cell.visible_edges = ""

    for col in range(n_cols):
        tbl[0, col].visible_edges = "TB"
        tbl[0, col].set_linewidth(1.4)
        tbl[n_rows, col].visible_edges = "B"
        tbl[n_rows, col].set_linewidth(1.4)

    plt.title("Table X. Summary of H3 Mediation Analyses (Theta, Beta, Alpha)",
               fontsize=11, fontweight="bold", pad=18, loc="left")
    plt.tight_layout()
    png_path = output_dir / "h3_results_table.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    Saved simplified H3 results table (image) -> {png_path}")
    return df


def _extract_full_channel_band_power(recall_dir, band, ch_names_wanted):
    """
    Like _extract_band_power, but returns power PER CHANNEL (averaged over
    epochs and time) instead of averaged down to one frontal scalar. Used to
    build the stress/recall correlation topomaps below.

    Returns dict {subj_num: {ch_name: power_scalar}}.
    """
    out = {}
    if not recall_dir.exists():
        return out
    for subj_dir in sorted(recall_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        band_path = subj_dir / "band_data.npz"
        meta_path = subj_dir / "cwt_metadata.npz"
        if not band_path.exists() or not meta_path.exists():
            continue
        subj_num = get_participant_num(subj_dir)
        if subj_num is None:
            continue
        try:
            bdata = np.load(band_path, allow_pickle=True)
            if band not in bdata:
                continue
            band_arr = np.asarray(bdata[band], dtype=np.float32)
            if band_arr.ndim == 3:
                band_arr = band_arr.mean(axis=0)  # avg over epochs -> (n_ch, n_time)
            band_arr = band_arr.mean(axis=1)      # avg over time -> (n_ch,)
            bdata.close()

            meta = np.load(meta_path, allow_pickle=True)
            ch_names = list(meta["ch_names"])
            per_ch = {}
            for ch in ch_names_wanted:
                if ch in ch_names:
                    per_ch[ch] = float(band_arr[ch_names.index(ch)])
            if per_ch:
                out[subj_num] = per_ch
        except Exception as e:
            print(f"    WARNING: Failed to extract per-channel {band} for {subj_num}: {e}")
    return out


def plot_band_power_by_condition_topomaps(recall_dir, recall_name, participant_metadata, output_dir):
    """
    One figure per recall session (not one file per band/condition):
    rows = bands (theta/alpha/beta), columns = conditions (control,
    ENG-SWA, SWA-ENG). Each cell is a topomap of average signal amplitude
    (uV) across subjects in that condition, at every scalp channel.

    This is the descriptive counterpart to plot_h3_eeg_topomaps(): that
    function shows WHERE power correlates with stress/recall; this one
    just shows the raw amplitude pattern itself, split by condition, so
    it can be visually compared to the numeric table from
    describe_band_power_by_condition().
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Channel positions/info, same approach as plot_h3_eeg_topomaps.
    info = None
    for subj_dir in sorted(recall_dir.iterdir()) if recall_dir.exists() else []:
        ep_path = subj_dir / "epochs_clean-epo.fif"
        if ep_path.exists():
            try:
                epochs = mne.read_epochs(ep_path, verbose=False)
                _, info = _get_eeg_picks_and_info(epochs)
                del epochs
                break
            except Exception:
                continue
    if info is None:
        print("    WARNING: could not find any epochs_clean-epo.fif to get channel "
              f"positions for condition topomaps ({recall_name}) - skipping.")
        return None

    eeg_ch_names = info["ch_names"]
    bands = BANDS

    fig, axes = plt.subplots(len(bands), len(CONDITIONS),
                              figsize=(5 * len(CONDITIONS), 4.3 * len(bands)))
    if len(bands) == 1:
        axes = axes.reshape(1, -1)

    for row, band in enumerate(bands):
        power_by_subj = _extract_full_channel_band_power(recall_dir, band, eeg_ch_names)

        # Build subject x channel matrix per condition, converting V -> uV.
        cond_matrices = {}
        for condition in CONDITIONS:
            rows_uv = []
            for subj_num, meta in participant_metadata.items():
                if meta.get("condition") != condition:
                    continue
                ch_power = power_by_subj.get(subj_num)
                if ch_power is None or not all(ch in ch_power for ch in eeg_ch_names):
                    continue
                rows_uv.append([ch_power[ch] * 1e6 for ch in eeg_ch_names])
            cond_matrices[condition] = np.array(rows_uv) if rows_uv else None

        # One shared color scale across all three conditions for this band,
        # so columns are visually comparable (same logic as
        # plot_topomaps_per_band's fix for per-frame rescaling).
        all_means = [m.mean(axis=0) for m in cond_matrices.values() if m is not None]
        if not all_means:
            for col in range(len(CONDITIONS)):
                axes[row, col].axis("off")
                axes[row, col].set_title(f"{band.upper()}\n(no data)")
            continue
        band_absmax = float(np.nanmax(np.concatenate(all_means))) or 1e-3
        vlim = (0.0, band_absmax)

        for col, condition in enumerate(CONDITIONS):
            mat = cond_matrices[condition]
            if mat is None or len(mat) < 3:
                axes[row, col].axis("off")
                axes[row, col].set_title(f"{band.upper()} - {condition}\n(insufficient data)")
                continue

            mean_uv = mat.mean(axis=0)
            mne.viz.plot_topomap(
                mean_uv, info, axes=axes[row, col],
                vlim=vlim, cmap="viridis", contours=6,
                sphere=0.095, extrapolate="head", show=False,
            )
            axes[row, col].set_title(
                f"{band.upper()} - {condition}\n(n={len(mat)}, mean={mean_uv.mean():.2f} uV)",
                fontsize=10,
            )

    plt.suptitle(f"Band power (uV) by condition - {recall_name}",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()

    plot_path = output_dir / f"band_power_by_condition_topomaps_{recall_name}.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved condition topomaps -> {plot_path}")
    return plot_path


def make_band_power_by_condition_table(recall1_csv, recall2_csv, output_path):
    """
    Build one clean summary table image (mean +/- SD in uV, plus the
    Kruskal-Wallis H/p) from the two CSVs saved by
    describe_band_power_by_condition(). Converts V -> uV so the table
    reads in normal EEG units instead of scientific notation.
    """
    df = pd.concat([pd.read_csv(recall1_csv), pd.read_csv(recall2_csv)], ignore_index=True)

    conds = df[df.condition != "KRUSKAL_WALLIS_ACROSS_CONDITIONS"].copy()
    kw = df[df.condition == "KRUSKAL_WALLIS_ACROSS_CONDITIONS"].copy()
    kw = kw.rename(columns={"mean": "H", "sd": "p"})[["recall", "band", "H", "p"]]

    band_order = ["theta", "alpha", "beta"]
    recall_order = ["recall1", "recall2"]
    cond_order = [c for c in CONDITIONS]

    rows = []
    for recall in recall_order:
        for band in band_order:
            row = {"Recall": "Immediate" if recall == "recall1" else "Delayed",
                   "Band": band.capitalize()}
            for cond in cond_order:
                sub = conds[(conds.recall == recall) & (conds.band == band) & (conds.condition == cond)]
                if len(sub):
                    m_uv = sub.iloc[0]["mean"] * 1e6
                    sd_uv = sub.iloc[0]["sd"] * 1e6
                    row[cond] = f"{m_uv:.2f}\n({sd_uv:.2f})"
                else:
                    row[cond] = "-"
            kwrow = kw[(kw.recall == recall) & (kw.band == band)]
            if len(kwrow):
                row["H(2)"] = f"{kwrow.iloc[0]['H']:.2f}"
                row["p"] = f"{kwrow.iloc[0]['p']:.3f}"
            rows.append(row)

    table_df = pd.DataFrame(rows)
    col_labels = ["Recall", "Band"] + cond_order + ["H(2)", "p"]
    table_df = table_df[col_labels]

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.axis("off")
    tbl = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.6)

    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor("#888888")
        if row_idx == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f7f7f7" if row_idx % 2 == 0 else "white")

    plt.title("Frontal-midline (Fz/FCz) band power (uV) by condition, per recall session\n"
               "Mean (SD); Kruskal-Wallis test across conditions",
               fontsize=11, weight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved table -> {output_path}")
    return output_path


def plot_h3_eeg_topomaps(recall_dir, output_dir, bands=("theta", "alpha", "beta")):
    """
    THE "EEG part" of H3, visually: for each band, correlate each channel's
    power across subjects with (1) stress and (2) delayed recall accuracy,
    then plot the resulting r-value at every scalp location as a topomap.

    This is different from (and complements) the plain grand-average band
    topomaps already produced by plot_topomaps_per_band(): those show WHERE
    on the scalp a band is strongest; these show WHERE on the scalp that
    band's power actually tracks stress/recall - i.e. a spatial picture of
    the H3 mediation pathway instead of just the two frontal channels used
    in the numeric test.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Grab one subject's epochs purely for channel positions/info (montage).
    info = None
    for subj_dir in sorted(recall_dir.iterdir()) if recall_dir.exists() else []:
        ep_path = subj_dir / "epochs_clean-epo.fif"
        if ep_path.exists():
            try:
                epochs = mne.read_epochs(ep_path, verbose=False)
                _, info = _get_eeg_picks_and_info(epochs)
                del epochs
                break
            except Exception:
                continue
    if info is None:
        print("    WARNING: could not find any epochs_clean-epo.fif to get channel "
              "positions for H3 topomaps - skipping.")
        return None

    eeg_ch_names = info["ch_names"]

    fig, axes = plt.subplots(2, len(bands), figsize=(5 * len(bands), 9))
    if len(bands) == 1:
        axes = axes.reshape(2, 1)

    for col, band in enumerate(bands):
        power_by_subj = _extract_full_channel_band_power(recall_dir, band, eeg_ch_names)

        # Build subject x channel matrix, keeping only subjects with complete
        # stress + recall2 + full channel coverage.
        subj_ids, stress_vals, recall_vals, power_rows = [], [], [], []
        for subj_num, meta in PARTICIPANT_METADATA.items():
            acc_r2 = meta.get("accuracy_recall2")
            stress_num = meta.get("stress")
            ch_power = power_by_subj.get(subj_num)
            if acc_r2 is None or stress_num is None or ch_power is None:
                continue
            if not all(ch in ch_power for ch in eeg_ch_names):
                continue
            subj_ids.append(subj_num)
            stress_vals.append(stress_num)
            recall_vals.append(float(acc_r2))
            power_rows.append([ch_power[ch] for ch in eeg_ch_names])

        if len(subj_ids) < 5:
            for row in range(2):
                axes[row, col].axis("off")
                axes[row, col].set_title(f"{band.upper()}\n(insufficient data)")
            continue

        power_mat = np.array(power_rows)          # (n_subj, n_ch)
        stress_vals = np.array(stress_vals)
        recall_vals = np.array(recall_vals)

        # Spearman for the same reason as the bar chart / averaged frontal
        # test: theta/alpha/beta power and recall are all significantly
        # right-skewed (Shapiro-Wilk), and Pearson vs Spearman gave a
        # different significance conclusion on the c-path in this dataset.
        r_stress = np.array([
            scipy_stats.spearmanr(power_mat[:, i], stress_vals)[0]
            for i in range(power_mat.shape[1])
        ])
        r_recall = np.array([
            scipy_stats.spearmanr(power_mat[:, i], recall_vals)[0]
            for i in range(power_mat.shape[1])
        ])

        for row, (r_vals, rel_label) in enumerate([
            (r_stress, "Stress <-> Power"), (r_recall, "Power <-> Delayed Recall")
        ]):
            absmax = float(np.nanmax(np.abs(r_vals))) or 0.01
            mne.viz.plot_topomap(
                r_vals, info, axes=axes[row, col],
                vlim=(-absmax, absmax), cmap="RdBu_r", contours=6,
                # FIX: sphere="auto" fits the sphere radius to this cap's
                # digitized positions, but MNE draws the nose/ear patches at
                # a size proportioned for the *standard* head radius
                # (~0.095 m). If the digitized cap isn't perfectly spherical,
                # "auto" can return a radius that no longer matches those
                # patches -> oversized nose/ears relative to the head
                # circle. Pinning sphere to the standard radius keeps the
                # patches and the head circle in the same proportion.
                sphere=0.095, extrapolate="head", show=False,
            )
            axes[row, col].set_title(
                f"{band.upper()}: {rel_label}\n(n={len(subj_ids)}, r range "
                f"[{r_vals.min():.2f}, {r_vals.max():.2f}])",
                fontsize=10,
            )

    plt.suptitle("H3: Scalp Correlation Maps, Spearman rho (stress-power and power-recall, per band)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = output_dir / "h3_stress_power_recall_topomaps.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved H3 EEG topomaps -> {plot_path}")
    return plot_path


# Rough scalp-region lookup by standard 10-20 channel-name prefix. Used only
# to order/colour bars in plot_h3_channel_bars below - it's a labelling
# convenience, not a statistical grouping (no averaging across channels).
_REGION_ORDER = ["Frontal", "Fronto-central", "Central", "Centro-parietal",
                  "Parietal", "Occipital", "Temporal", "Other"]
_REGION_COLORS = {
    "Frontal": "#4C72B0", "Fronto-central": "#64A6BD", "Central": "#55A868",
    "Centro-parietal": "#C4B454", "Parietal": "#DD8452", "Occipital": "#C44E52",
    "Temporal": "#8172B2", "Other": "#999999",
}


def _channel_region(ch_name):
    name = ch_name.upper()
    if name.startswith("FP") or name.startswith("AF"):
        return "Frontal"
    if name.startswith("FC") or name.startswith("FT"):
        return "Fronto-central"
    if name.startswith("F"):
        return "Frontal"
    if name.startswith("CP") or name.startswith("TP"):
        return "Centro-parietal"
    if name.startswith("C"):
        return "Central"
    if name.startswith("PO"):
        return "Parietal"
    if name.startswith("P"):
        return "Parietal"
    if name.startswith("O"):
        return "Occipital"
    if name.startswith("T"):
        return "Temporal"
    return "Other"


def plot_h3_channel_bars(recall_dir, output_dir, bands=("theta", "alpha", "beta")):
    """
    Alternative to the H3 scalp topomaps: a per-channel horizontal bar chart
    of the same stress<->power and power<->recall correlations, one panel
    per band (2 rows x len(bands) cols, matching the topomap layout).

    Why a bar chart instead of a topomap here:
      - None of the channel-wise correlations in this dataset reach the
        n=61 significance threshold (|r| >= .25, dashed reference line
        below). A smoothed/interpolated topomap visually implies a
        continuous spatial gradient *between* electrodes and can read as a
        localized "hot spot" even when the underlying values are
        essentially noise and were not corrected for the number of
        channels tested. A bar chart shows only the values that were
        actually measured, at the resolution they were actually measured,
        with no interpolation.
      - Exact r-values are readable directly (useful for a thesis
        appendix), whereas colour alone under-informs the reader.
    Bars are coloured by scalp region (label only, not a statistical
    grouping) and sorted by region then by channel name so the frontal
    theta channels central to H3 (Fz, FCz) are easy to find.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    info = None
    for subj_dir in sorted(recall_dir.iterdir()) if recall_dir.exists() else []:
        ep_path = subj_dir / "epochs_clean-epo.fif"
        if ep_path.exists():
            try:
                epochs = mne.read_epochs(ep_path, verbose=False)
                _, info = _get_eeg_picks_and_info(epochs)
                del epochs
                break
            except Exception:
                continue
    if info is None:
        print("    WARNING: could not find any epochs_clean-epo.fif to get channel "
              "names for H3 channel bar chart - skipping.")
        return None

    eeg_ch_names = info["ch_names"]
    # Order channels by region (for consistent, readable bar grouping)
    order = sorted(range(len(eeg_ch_names)),
                    key=lambda i: (_REGION_ORDER.index(_channel_region(eeg_ch_names[i])),
                                    eeg_ch_names[i]))
    ch_sorted = [eeg_ch_names[i] for i in order]
    ch_colors = [_REGION_COLORS[_channel_region(ch)] for ch in ch_sorted]

    SIG_THRESH = 0.25  # critical r at n=61, matches the value reported in-text

    fig, axes = plt.subplots(2, len(bands), figsize=(4.5 * len(bands), 10), sharex=True)
    if len(bands) == 1:
        axes = axes.reshape(2, 1)

    for col, band in enumerate(bands):
        power_by_subj = _extract_full_channel_band_power(recall_dir, band, eeg_ch_names)

        subj_ids, stress_vals, recall_vals, power_rows = [], [], [], []
        for subj_num, meta in PARTICIPANT_METADATA.items():
            acc_r2 = meta.get("accuracy_recall2")
            stress_num = meta.get("stress")
            ch_power = power_by_subj.get(subj_num)
            if acc_r2 is None or stress_num is None or ch_power is None:
                continue
            if not all(ch in ch_power for ch in eeg_ch_names):
                continue
            subj_ids.append(subj_num)
            stress_vals.append(stress_num)
            recall_vals.append(float(acc_r2))
            power_rows.append([ch_power[ch] for ch in eeg_ch_names])

        if len(subj_ids) < 5:
            for row in range(2):
                axes[row, col].axis("off")
                axes[row, col].set_title(f"{band.upper()}\n(insufficient data)")
            continue

        power_mat = np.array(power_rows)[:, order]  # reorder columns to match ch_sorted
        stress_vals = np.array(stress_vals)
        recall_vals = np.array(recall_vals)

        # Spearman, not Pearson: theta/alpha/beta power and recall were all
        # significantly right-skewed under Shapiro-Wilk (reported in-text),
        # and the c-path result (Pearson n.s. vs Spearman p=.035) showed
        # this isn't a cosmetic distinction - it can flip a conclusion.
        # Spearman is used here per-channel for the same reason it was
        # treated as the more trustworthy estimate for the averaged
        # frontal metric.
        r_stress = np.array([scipy_stats.spearmanr(power_mat[:, i], stress_vals)[0]
                              for i in range(power_mat.shape[1])])
        r_recall = np.array([scipy_stats.spearmanr(power_mat[:, i], recall_vals)[0]
                              for i in range(power_mat.shape[1])])

        for row, (r_vals, rel_label) in enumerate([
            (r_stress, "Stress <-> Power"), (r_recall, "Power <-> Delayed Recall")
        ]):
            ax = axes[row, col]
            y_pos = np.arange(len(ch_sorted))
            ax.barh(y_pos, r_vals, color=ch_colors, edgecolor="white", height=0.75)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.axvline(SIG_THRESH, color="grey", linewidth=1, linestyle="--")
            ax.axvline(-SIG_THRESH, color="grey", linewidth=1, linestyle="--")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(ch_sorted, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlim(-0.4, 0.4)
            ax.set_title(f"{band.upper()}: {rel_label}\n(n={len(subj_ids)})", fontsize=10)
            if row == 1:
                ax.set_xlabel("Spearman ρ")

    # Region legend (shared)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_REGION_COLORS[r]) for r in _REGION_ORDER
               if r in {_channel_region(c) for c in eeg_ch_names}]
    labels = [r for r in _REGION_ORDER if r in {_channel_region(c) for c in eeg_ch_names}]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    plt.suptitle("H3: Per-Channel Spearman Correlations (stress-power and power-recall, per band)\n"
                  "dashed line = critical |rho| for significance at n=61 (>= .25)",
                  fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plot_path = output_dir / "h3_stress_power_recall_channel_bars.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved H3 per-channel bar chart -> {plot_path}")
    return plot_path


def run_hypothesis_tests(recall_dir, output_dir):
    """
    Run all hypothesis tests (H1, H2, recall1-vs-recall2 change score, H3, H3b).

    recall_dir: path to EEG_output directory containing "recall1"/"recall2" subfolders
    output_dir: path to save results (EEG_output/hypothesis_tests)

    H1 and H2 are run on BOTH recall1 and recall2:
      - recall2 (the delayed test) is the PRIMARY hypothesis test.
      - recall1 is a comparison/control check - is there already a group
        difference at recall1, before the delay/retrieval-practice
        manipulation had a chance to matter?
    A paired recall1->recall2 change-score analysis is also run per
    condition, which is the direct test of whether retrieval practice
    caused a bigger GAIN over time (not just a higher recall2 endpoint).
    """
    print(f"\n{'#'*60}")
    print("# HYPOTHESIS TESTING PIPELINE")
    print(f"{'#'*60}")

    tests_dir = output_dir / "hypothesis_tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    recall2_dir = recall_dir / "recall2"
    recall1_dir = recall_dir / "recall1"

    def _build_recall_dict(acc_key):
        data = {}
        for subj_num, meta in PARTICIPANT_METADATA.items():
            acc = meta.get(acc_key)
            if acc is not None:
                data[subj_num] = acc
        return data

    recall1_data = _build_recall_dict("accuracy_recall1")
    recall2_data = _build_recall_dict("accuracy_recall2")

    print(f"\nLoaded recall1 accuracy for {len(recall1_data)} participants from metadata.")
    print(f"Loaded recall2 accuracy for {len(recall2_data)} participants from metadata.")

    # H1 - primary test on recall2, comparison check on recall1
    h1_recall2 = test_h1_retrieval_practice(recall2_data, tests_dir, recall_col="recall2")
    h1_recall1 = test_h1_retrieval_practice(recall1_data, tests_dir, recall_col="recall1")

    # H2 - primary test on recall2, comparison check on recall1
    h2_recall2 = test_h2_language_order_interaction(recall2_data, tests_dir, recall_col="recall2")
    h2_recall1 = test_h2_language_order_interaction(recall1_data, tests_dir, recall_col="recall1")

    # -- Figures: colorful boxplots + tables for H1/H2, same visual style as
    #    the Figure 6 boxplot and the significance heatmap you already have.
    figures_dir = tests_dir / "figures"
    plot_recall_boxplots_by_condition(figures_dir)
    plot_h2_boxplots(figures_dir)
    plot_h1_h2_significance_heatmap(h1_recall1, h1_recall2, h2_recall1, h2_recall2, figures_dir)
    build_h1_h2_results_table(h1_recall1, h1_recall2, h2_recall1, h2_recall2, figures_dir)

    # Recall1 -> Recall2 change score (paired, per condition) - the direct
    # test of whether retrieval practice caused a bigger GAIN over time.
    change_score = test_recall_change_score(tests_dir)

    # H3 (needs EEG data; stress is now a continuous covariate)
    h3 = test_h3_stress_theta_recall(recall2_dir if recall2_dir.exists() else recall_dir, recall2_data, tests_dir)

    # H3b - exploratory beta-band companion (opposite predicted direction to theta)
    h3b = test_h3b_stress_beta_recall(recall2_dir if recall2_dir.exists() else recall_dir, recall2_data, tests_dir)

    # H3c - exploratory alpha-band companion (same predicted direction as theta)
    h3c = test_h3c_stress_alpha_recall(recall2_dir if recall2_dir.exists() else recall_dir, recall2_data, tests_dir)

    # -- H3 figures: one consolidated theta/beta/alpha mediation table, plus
    #    scalp correlation topomaps showing WHERE stress<->power and
    #    power<->recall associations are strongest.
    build_h3_results_table(h3, h3b, h3c, figures_dir)
    # Primary recommendation: per-channel bar chart (no spatial interpolation,
    # honest about the null/non-significant channel-wise correlations).
    plot_h3_channel_bars(recall2_dir if recall2_dir.exists() else recall_dir, figures_dir)
    # Kept as an optional supplementary visual, proportions fixed (sphere=0.095).
    plot_h3_eeg_topomaps(recall2_dir if recall2_dir.exists() else recall_dir, figures_dir)

    # Summary
    print(f"\n{'='*60}")
    print("HYPOTHESIS TEST SUMMARY")
    print("="*60)
    print(f"  H1 (Retrieval Practice, recall2 - PRIMARY): "
          f"{'SUPPORTED' if h1_recall2 and h1_recall2.get('h1_supported') else 'NOT SUPPORTED'}")
    print(f"  H1 (Retrieval Practice, recall1 - control check): "
          f"{'group difference present' if h1_recall1 and h1_recall1.get('h1_supported') else 'no group difference'}")
    print(f"  H2 (Language Order, recall2 - PRIMARY):     "
          f"{'SUPPORTED' if h2_recall2 and h2_recall2.get('interaction_significant') else 'NOT SUPPORTED'}")
    print(f"  H2 (Language Order, recall1 - control check): "
          f"{'interaction present' if h2_recall1 and h2_recall1.get('interaction_significant') else 'no interaction'}")
    print(f"  H3 (Stress-Theta-Recall): {'SUPPORTED' if h3 and h3.get('h3_supported') else 'NOT SUPPORTED'}")
    if h3b:
        print(f"  H3b (Stress-Beta-Recall, exploratory): r={h3b.get('stress_beta_r')}, "
              f"direction {'matches' if h3b.get('direction_matches_literature') else 'does not match'} literature")
    if h3c:
        print(f"  H3c (Stress-Alpha-Recall, exploratory): r={h3c.get('stress_alpha_r')}, "
              f"direction {'matches' if h3c.get('direction_matches_literature') else 'does not match'} literature")

    # Save combined summary
    def _h1_summary(h1):
        if not h1:
            return {"supported": None, "p_value": None}
        anova_p = h1.get("anova", {}).get("p")
        return {"supported": h1.get("h1_supported"), "anova_p": anova_p,
                "pairwise": h1.get("pairwise_welch_bonferroni")}

    def _h2_summary(h2):
        if not h2:
            return {"supported": None, "p_value": None}
        return {"supported": h2.get("interaction_significant"),
                "p_value": h2.get("bootstrap_p_interaction")}

    summary = {
        "h1_recall2_primary": _h1_summary(h1_recall2),
        "h1_recall1_control_check": _h1_summary(h1_recall1),
        "h2_recall2_primary": _h2_summary(h2_recall2),
        "h2_recall1_control_check": _h2_summary(h2_recall1),
        "recall_change_score": {
            "anova_p": change_score.get("change_score_anova", {}).get("p") if change_score else None,
            "kruskal_p": change_score.get("change_score_kruskal", {}).get("p") if change_score else None,
        },
        "h3": {"supported": h3 and h3.get("h3_supported"), "p_values": {
            "stress_theta": h3.get("stress_theta_p") if h3 else None,
            "theta_recall": h3.get("theta_recall_p") if h3 else None,
            "stress_recall": h3.get("stress_recall_p") if h3 else None,
        }},
        "h3b_beta_exploratory": {
            "direction_matches_literature": h3b.get("direction_matches_literature") if h3b else None,
            "boot_significant": h3b.get("mediation", {}).get("boot_significant") if h3b else None,
            "p_values": {
                "stress_beta": h3b.get("stress_beta_p") if h3b else None,
                "beta_recall": h3b.get("beta_recall_p") if h3b else None,
                "stress_recall": h3b.get("stress_recall_p") if h3b else None,
            },
        },
        "h3c_alpha_exploratory": {
            "direction_matches_literature": h3c.get("direction_matches_literature") if h3c else None,
            "boot_significant": h3c.get("mediation", {}).get("boot_significant") if h3c else None,
            "p_values": {
                "stress_alpha": h3c.get("stress_alpha_p") if h3c else None,
                "alpha_recall": h3c.get("alpha_recall_p") if h3c else None,
                "stress_recall": h3c.get("stress_recall_p") if h3c else None,
            },
        },
    }
    with open(tests_dir / "hypothesis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary -> {tests_dir / 'hypothesis_summary.json'}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # -- EEG PROCESSING PIPELINE - commented out, files already saved to disk --
    # Uncomment this block only if you need to reprocess the raw EEG files.
    #
    # print(f"Found {len(ALL_FILES)} files")
    # recall1_files, recall2_files = group_by_recall(ALL_FILES)
    # print(f"  recall1: {len(recall1_files)} files")
    # print(f"  recall2: {len(recall2_files)} files")
    #
    # for recall_name, files in [("recall1", recall1_files), ("recall2", recall2_files)]:
    #     if not files:
    #         print(f"\n  No files for {recall_name} - skipping.")
    #         continue
    #
    #     recall_dir = OUTPUT_ROOT / recall_name
    #     recall_dir.mkdir(parents=True, exist_ok=True)
    #
    #     cond_accum      = {c: {b: [] for b in BANDS} for c in CONDITIONS}
    #     cond_rep_epochs = {}
    #     # NOTE: stress is now a continuous covariate (see H3/H3b), so it is no
    #     # longer used to bucket subjects into topomap groups here.
    #
    #     for file in files:
    #         meta = get_metadata(file)
    #         if meta is None:
    #             continue
    #         condition = meta["condition"]
    #         if condition not in CONDITIONS:
    #             print(f"    WARNING: Unknown condition '{condition}' for {file.name} - skipping.")
    #             continue
    #         subj_dir = recall_dir / Path(file).stem
    #         try:
    #             print(f"\nProcessing [{recall_name}] {file.name}  (condition={condition})")
    #             band_data = run_pipeline(str(file), output_root=recall_dir)
    #             for band_name, band_arr in band_data.items():
    #                 cond_accum[condition][band_name].append(band_arr)
    #             ep_path = subj_dir / "epochs_clean-epo.fif"
    #             if ep_path.exists():
    #                 if condition not in cond_rep_epochs:
    #                     cond_rep_epochs[condition] = mne.read_epochs(ep_path, verbose=False)
    #         except Exception as e:
    #             print(f"\nERROR processing {file.name}: {e}")
    #             import traceback; traceback.print_exc()
    #
    #     run_analysis_for_groups(cond_accum, recall_name, cond_rep_epochs, "condition")
    #     run_behavioral_analysis(recall_dir, recall_name)

    # -- ANALYSIS ONLY - reads from already-saved files, no EEG reprocessing --
    # Regenerate topomaps/time-courses/TF tables for each recall using the
    # already-saved band_data.npz + epochs files (uses the fixed
    # plot_topomaps_per_band - global per-band color scale, sequential
    # colormap for amplitude/power, extrapolate='local'). This does NOT
    # touch the raw .vhdr files, so it's safe to run repeatedly.
    for _recall_name in ("recall1", "recall2"):
        _recall_dir = OUTPUT_ROOT / _recall_name
        if _recall_dir.exists():
            regenerate_topomaps_from_saved(_recall_dir, _recall_name, OUTPUT_ROOT)

    # -- Descriptive check: band power by condition, per recall session --
    # Not tied to a hypothesis - just reports whether theta/alpha/beta power
    # (at Fz/FCz) look different across control/ENG-SWA/SWA-ENG.
    _participant_metadata = load_participant_metadata()
    _band_power_csvs = {}
    for _recall_name in ("recall1", "recall2"):
        _recall_dir = OUTPUT_ROOT / _recall_name
        if _recall_dir.exists():
            describe_band_power_by_condition(_recall_dir, _recall_name,
                                              _participant_metadata, OUTPUT_ROOT)
            _band_power_csvs[_recall_name] = OUTPUT_ROOT / f"band_power_by_condition_{_recall_name}.csv"

            # One combined figure per recall session (bands x conditions),
            # instead of a separate file per band/condition.
            plot_band_power_by_condition_topomaps(_recall_dir, _recall_name,
                                                   _participant_metadata, OUTPUT_ROOT)

    # Single summary table (both recall sessions, all bands), in uV.
    if "recall1" in _band_power_csvs and "recall2" in _band_power_csvs:
        make_band_power_by_condition_table(
            _band_power_csvs["recall1"], _band_power_csvs["recall2"],
            OUTPUT_ROOT / "band_power_by_condition_table.png",
        )

    run_behavioral_analysis(OUTPUT_ROOT / "recall2", "recall2")
    run_hypothesis_tests(OUTPUT_ROOT, OUTPUT_ROOT)

    print("\nOK Pipeline complete.")
