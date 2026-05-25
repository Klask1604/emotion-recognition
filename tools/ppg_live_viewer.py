#!/usr/bin/env python3
"""
PPG Live Viewer — Galaxy Watch 7.

Update imediat la fiecare batch primit (event-driven, nu timer fix).
Fereastra scrolling 30s, semnal raw + filtrat + vârfuri + IBI + metrici HRV.

Rulare:
    pip install matplotlib scipy numpy paho-mqtt
    python ppg_live_viewer.py
"""

import argparse
import json
import threading
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # schimba in "Qt5Agg" daca TkAgg nu e disponibil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import signal
import paho.mqtt.client as mqtt

# ── Config ────────────────────────────────────────────────────────────────────
BROKER      = "paxbespoke.automateflow.ro"
PORT        = 1883
FS          = 25
DISPLAY_SEC = 30
BUFFER_SEC  = 120
SOS         = signal.butter(4, [0.5 / 12.5, 4.0 / 12.5], btype="band", output="sos")

# ── State thread-safe ─────────────────────────────────────────────────────────
_lock = threading.Lock()
_green_buf: deque[int] = deque(maxlen=FS * BUFFER_SEC)
_ts_buf:    deque[int] = deque(maxlen=FS * BUFFER_SEC)
_hrv:       dict       = {}
_new_data   = threading.Event()      # semnalizeaza ca a sosit un batch nou
_last_batch_time = [0.0]
_last_batch_n    = [0]
_last_batch_span = [0]
_total_batches   = [0]

# ── MQTT ──────────────────────────────────────────────────────────────────────
def _on_connect(client, ud, flags, rc, props=None):
    if rc == 0:
        print("[OK] Conectat")
        client.subscribe("biofizic/ppg/raw", qos=0)
        client.subscribe("biofizic/ppg_hrv", qos=0)
    else:
        print(f"[ERR] rc={rc}")


def _on_message(client, ud, msg):
    try:
        data = json.loads(msg.payload.decode("utf-8", errors="replace"))
    except Exception:
        return

    if msg.topic == "biofizic/ppg/raw":
        ts_list    = data.get("ts",    [])
        green_list = data.get("green", [])
        if not ts_list or not green_list:
            return
        with _lock:
            for ts, g in zip(ts_list, green_list):
                _ts_buf.append(int(ts))
                _green_buf.append(int(g))
            _last_batch_time[0] = time.time()
            _last_batch_n[0]    = len(ts_list)
            _last_batch_span[0] = (ts_list[-1] - ts_list[0]) if len(ts_list) > 1 else 0
            _total_batches[0]  += 1
        _new_data.set()          # trezeste bucla de plot

    elif msg.topic == "biofizic/ppg_hrv":
        with _lock:
            _hrv.update(data)
        _new_data.set()


# ── Culori ────────────────────────────────────────────────────────────────────
BG    = "#0d1117"
PANEL = "#161b22"
GRID  = "#21262d"
C_RAW  = "#58a6ff"
C_FILT = "#3fb950"
C_PEAK = "#f85149"
C_IBI  = "#e3b341"
C_DIM  = "#8b949e"
C_TXT  = "#e6edf3"


def _style(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=C_DIM, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    if title:
        ax.set_title(title, color=C_TXT, fontsize=8.5, pad=4)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.8)


# ── Build figure ──────────────────────────────────────────────────────────────
plt.rcParams.update({"toolbar": "None"})
fig = plt.figure(figsize=(15, 9), facecolor=BG)
fig.suptitle("PPG Live Viewer — Galaxy Watch 7", color=C_TXT, fontsize=11,
             fontweight="bold", y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.3,
                       top=0.94, bottom=0.06, left=0.05, right=0.97)

ax_raw  = fig.add_subplot(gs[0, :])
ax_filt = fig.add_subplot(gs[1, :])
ax_ibi  = fig.add_subplot(gs[2, 0:2])
ax_info = fig.add_subplot(gs[2, 2])

for ax in (ax_raw, ax_filt, ax_ibi):
    _style(ax)

ax_info.set_facecolor(PANEL)
ax_info.axis("off")
for sp in ax_info.spines.values():
    sp.set_color(GRID)
ax_info.set_title("Ultima epocă PPG", color=C_TXT, fontsize=8.5, pad=4)

line_raw,  = ax_raw.plot([], [],  color=C_RAW,  lw=0.7)
line_filt, = ax_filt.plot([], [], color=C_FILT, lw=0.8)
peaks_plt, = ax_filt.plot([], [], "v", color=C_PEAK, ms=5, zorder=5)

ax_raw.set_ylabel("ADC",  color=C_DIM, fontsize=7)
ax_filt.set_ylabel("amp", color=C_DIM, fontsize=7)

info_txt = ax_info.text(
    0.07, 0.96, "Astept ppg_hrv\n(~30s prima epocă)...",
    transform=ax_info.transAxes, color=C_TXT, fontsize=9,
    va="top", fontfamily="monospace", linespacing=1.8,
)

# Indicator "batch primit" (dreptunghi flash)
_flash_patch = FancyBboxPatch(
    (0.72, 0.945), 0.27, 0.045, transform=fig.transFigure,
    boxstyle="round,pad=0.003", facecolor=C_FILT, alpha=0.0, zorder=10,
)
fig.add_artist(_flash_patch)
_flash_txt = fig.text(
    0.855, 0.965, "", color=BG, fontsize=7.5, fontweight="bold",
    ha="center", va="center", zorder=11,
)

status_txt = fig.text(
    0.01, 0.01, "Astept date biofizic/ppg/raw...",
    color=C_DIM, fontsize=7.5, fontfamily="monospace",
)

_flash_alpha = [0.0]
_ibi_dirty   = [False]  # redraw IBI doar cand avem date noi


# ── Draw helper ───────────────────────────────────────────────────────────────
def _draw():
    with _lock:
        n_total = len(_green_buf)
        if n_total < 10:
            return
        n_disp = FS * DISPLAY_SEC
        green  = np.array(list(_green_buf)[-n_disp:], dtype=float)
        hrv    = dict(_hrv)
        bt     = _last_batch_time[0]
        bn     = _last_batch_n[0]
        bspan  = _last_batch_span[0]
        tot    = _total_batches[0]

    t = np.arange(len(green)) / float(FS)

    # ── Raw ────────────────────────────────────────────────────────────
    line_raw.set_data(t, green)
    ax_raw.set_xlim(0, DISPLAY_SEC)
    g_rng = float(np.ptp(green))
    if g_rng > 0:
        m = g_rng * 0.07
        ax_raw.set_ylim(green.min() - m, green.max() + m)
    n_sec = len(green) / FS
    ax_raw.set_title(
        f"Raw PPG  green channel  —  {n_sec:.0f}s acumulat  ({n_total} samples total)",
        color=C_TXT, fontsize=8.5, pad=4,
    )

    # ── Filtrat + peaks ────────────────────────────────────────────────
    peaks = np.array([], dtype=int)
    ibi_ok = np.array([])
    if len(green) >= FS * 2:
        filt = signal.sosfiltfilt(SOS, green)
        line_filt.set_data(t, filt)
        ax_filt.set_xlim(0, DISPLAY_SEC)
        f_rng = float(np.ptp(filt))
        if f_rng > 0:
            m = f_rng * 0.1
            ax_filt.set_ylim(filt.min() - m, filt.max() + m)

        min_dist = int(FS * 0.4)
        prom = max(0.01 * f_rng, 1e-9) if f_rng > 0 else 1e-9
        peaks, _ = signal.find_peaks(filt, distance=min_dist, prominence=prom)
        if len(peaks) > 0:
            peaks_plt.set_data(t[peaks], filt[peaks])
        else:
            peaks_plt.set_data([], [])

        if len(peaks) >= 2:
            ibi_ms = np.diff(peaks) / FS * 1000.0
            ibi_ok = ibi_ms[(ibi_ms >= 300) & (ibi_ms <= 2000)]

        hr_live = ""
        if len(ibi_ok) >= 2:
            mean_ibi = float(np.mean(ibi_ok))
            hr_live  = f"  HR≈{60000/mean_ibi:.0f}bpm"

        ax_filt.set_title(
            f"Filtrat (Butterworth 0.5–4 Hz)  {len(peaks)} vârfuri{hr_live}",
            color=C_TXT, fontsize=8.5, pad=4,
        )

    # ── IBI barchart ───────────────────────────────────────────────────
    if len(ibi_ok) >= 2 and _ibi_dirty[0]:
        _ibi_dirty[0] = False
        t_pk = t[peaks[:-1]][:len(ibi_ok)]
        ax_ibi.cla()
        _style(ax_ibi)
        colors = [C_IBI if 400 <= v <= 1500 else C_PEAK for v in ibi_ok]
        bw = max(0.25, DISPLAY_SEC / max(len(ibi_ok), 1) * 0.55)
        ax_ibi.bar(t_pk, ibi_ok, width=bw, color=colors, alpha=0.85,
                   edgecolor=PANEL, lw=0.5)
        m_ibi = float(np.mean(ibi_ok))
        ax_ibi.axhline(m_ibi, color="#ff7b72", lw=1, ls="--")
        ax_ibi.set_ylim(200, 1800)
        ax_ibi.set_xlim(0, DISPLAY_SEC)
        ax_ibi.set_xlabel("timp (s)", color=C_DIM, fontsize=7)
        ax_ibi.set_title(
            f"IBI (ms)  mean={m_ibi:.0f}ms  n={len(ibi_ok)}",
            color=C_TXT, fontsize=8.5, pad=4,
        )
        ax_ibi.tick_params(colors=C_DIM, labelsize=7)
        ax_ibi.grid(True, color=GRID, lw=0.5, alpha=0.8)

    # ── Metrici HRV ────────────────────────────────────────────────────
    if hrv:
        def f(k, d=1):
            v = hrv.get(k)
            return f"{v:.{d}f}" if isinstance(v, (int, float)) else "—"

        z = hrv.get("z_pulse_amp", 0.0)
        z_sym = ("▲ act" if isinstance(z, float) and z > 0.3 else
                 "▼ rel" if isinstance(z, float) and z < -0.3 else "● neutru")
        base = hrv.get("amp_baseline_ready", False)
        epoch = hrv.get("epoch_n", "—")

        info_txt.set_text(
            f"Epoca PPG #{epoch}\n"
            f"\n"
            f"HR     {f('mean_hr_ppg', 0):>7} bpm\n"
            f"RMSSD  {f('rmssd_ppg', 1):>7} ms\n"
            f"SDNN   {f('sdnn_ppg', 1):>7} ms\n"
            f"IBI n  {f('ibi_n_ppg', 0):>7}\n"
            f"\n"
            f"Amp    {f('pulse_amp_mean', 4)}\n"
            f"z_amp  {f('z_pulse_amp', 3)}  {z_sym}\n"
            f"Base   {'✓ gata' if base else '⟳ calibrare'}"
        )

    # ── Flash indicator ────────────────────────────────────────────────
    age = time.time() - bt if bt > 0 else 9999
    _flash_alpha[0] = max(0.0, _flash_alpha[0] - 0.08)
    _flash_patch.set_alpha(_flash_alpha[0])

    # ── Status bar ─────────────────────────────────────────────────────
    rate = bn / (bspan / 1000.0) if bspan > 0 else 0.0
    status_txt.set_text(
        f"Ultimul batch: {bn} samples  span={bspan}ms  "
        f"fs_real={rate:.1f}Hz  acum {age:.1f}s  |  "
        f"total {tot} batch-uri  |  buffer {n_total} samples ({n_total/FS:.0f}s)"
    )

    fig.canvas.draw_idle()


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default=BROKER)
    parser.add_argument("--port",   type=int, default=PORT)
    args = parser.parse_args()

    client = mqtt.Client(
        client_id="biofizic_ppg_live",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.connect(args.broker, args.port, keepalive=60)
    client.loop_start()

    plt.ion()
    plt.show(block=False)

    print("Fereastra deschisa. Astept date PPG (Ctrl+C sau inchide fereastra).")

    try:
        while plt.fignum_exists(fig.number):
            # Asteapta pana soseste un batch nou (max 1s inainte sa actualizezi statusul)
            got_data = _new_data.wait(timeout=1.0)
            if got_data:
                _new_data.clear()
                _ibi_dirty[0] = True
                # Flash vizual: dreptunghi verde aprins
                with _lock:
                    bn = _last_batch_n[0]
                _flash_alpha[0] = 0.85
                _flash_patch.set_alpha(0.85)
                _flash_txt.set_text(f"▶ batch  {bn} samples")
            # Fade flash chiar si fara date noi
            _flash_alpha[0] = max(0.0, _flash_alpha[0] - 0.06)
            _flash_patch.set_alpha(_flash_alpha[0])
            if _flash_alpha[0] < 0.05:
                _flash_txt.set_text("")

            _draw()
            plt.pause(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()
        print("Viewer oprit.")


if __name__ == "__main__":
    main()
