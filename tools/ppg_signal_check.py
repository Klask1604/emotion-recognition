#!/usr/bin/env python3
"""
Test 2 — Captureaza 30s de PPG raw si ploteza raw vs filtrat.

Ruleaza local (necesita matplotlib):
    pip install matplotlib scipy numpy paho-mqtt
    python ppg_signal_check.py

Salveaza ppg_signal.png in directorul curent.
Semnal bun: pulsuri clare la ~1Hz (la HR=60bpm = 1 puls/s).
"""

import json
import sys
import threading

import numpy as np
import paho.mqtt.client as mqtt
from scipy import signal

BROKER     = "paxbespoke.automateflow.ro"
PORT       = 1883
TARGET_S   = 30
FS         = 25
TARGET_N   = TARGET_S * FS  # 750 samples

green_all: list[int] = []
ts_all:    list[int] = []
done_evt = threading.Event()


def on_connect(client, userdata, flags, rc, props=None):
    if rc == 0:
        print(f"[OK] Conectat. Capturez {TARGET_S}s de PPG...")
        client.subscribe("biofizic/ppg/raw", qos=0)
    else:
        print(f"[ERR] Conectare esuata rc={rc}")
        done_evt.set()


def on_message(client, userdata, msg):
    if done_evt.is_set():
        return
    try:
        data = json.loads(msg.payload)
    except Exception:
        return
    green_all.extend(data.get("green", []))
    ts_all.extend(data.get("ts", []))
    n = len(green_all)
    if n % 50 == 0:
        print(f"  {n}/{TARGET_N} samples ({100*n//TARGET_N}%)", end="\r", flush=True)
    if n >= TARGET_N:
        done_evt.set()


def main():
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        print("[ERR] matplotlib nu e instalat: pip install matplotlib")
        sys.exit(1)

    c = mqtt.Client(
        client_id="biofizic_ppg_check",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    c.on_connect = on_connect
    c.on_message = on_message
    c.connect(BROKER, PORT, keepalive=60)
    c.loop_start()

    print(f"Astept {TARGET_S}s de date PPG raw (Ctrl+C pentru anulare)...")
    done_evt.wait(timeout=TARGET_S + 30)
    c.loop_stop()
    c.disconnect()

    if len(green_all) < FS * 10:
        print(f"\n[ERR] Prea putine samples: {len(green_all)}. Verifica ca ceasul trimite PPG raw.")
        sys.exit(1)

    green = np.array(green_all[:TARGET_N], dtype=float)
    ts    = np.array(ts_all[:TARGET_N])
    t     = np.arange(len(green)) / float(FS)

    # Verificare rate reala
    if len(ts) >= 2:
        actual_fs = (len(ts) - 1) * 1000.0 / (ts[-1] - ts[0])
        print(f"\n[INFO] Rata reala: {actual_fs:.1f} Hz (nominal: {FS} Hz)")
        if abs(actual_fs - FS) > 5:
            print(f"[WARN] Rata deviaza semnificativ. Verifica dp.timestamp pe ceas.")

    # Filtru Butterworth 0.5–4 Hz
    sos      = signal.butter(4, [0.5 / 12.5, 4.0 / 12.5], btype="band", output="sos")
    filtered = signal.sosfiltfilt(sos, green)

    # Detectie vârfuri
    min_dist = int(FS * 0.4)
    prom     = max(0.01 * float(np.ptp(filtered)), 1e-6)
    peaks, _ = signal.find_peaks(filtered, distance=min_dist, prominence=prom)
    print(f"[INFO] Vârfuri detectate: {len(peaks)} în {TARGET_S}s = ~{len(peaks)/TARGET_S*60:.0f} bpm")

    # IBI
    if len(peaks) >= 2:
        ibi_ms = np.diff(peaks) / FS * 1000.0
        print(f"[INFO] IBI: mean={np.mean(ibi_ms):.0f}ms, std={np.std(ibi_ms):.0f}ms, "
              f"range=[{np.min(ibi_ms):.0f},{np.max(ibi_ms):.0f}]ms")
        diffs = np.diff(ibi_ms)
        if len(diffs) >= 2:
            rmssd = float(np.sqrt(np.mean(diffs ** 2)))
            print(f"[INFO] RMSSD estimat: {rmssd:.1f}ms")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))

    axes[0].plot(t, green, linewidth=0.8, color="tab:blue")
    axes[0].set_title(f"PPG Raw (green channel) — {len(green)} samples @ {FS}Hz")
    axes[0].set_xlabel("timp (s)")
    axes[0].set_ylabel("ADC counts")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, filtered, linewidth=0.8, color="tab:green")
    if len(peaks) > 0:
        axes[1].plot(t[peaks], filtered[peaks], "rv", markersize=6, label=f"{len(peaks)} vârfuri")
        axes[1].legend()
    axes[1].set_title("PPG Filtrat (Butterworth 0.5–4 Hz)")
    axes[1].set_xlabel("timp (s)")
    axes[1].set_ylabel("amplitude normalizată")
    axes[1].grid(True, alpha=0.3)

    if len(peaks) >= 2:
        t_peaks = t[peaks[:-1]]
        axes[2].bar(t_peaks, ibi_ms, width=0.5, color="tab:orange", alpha=0.7)
        axes[2].axhline(np.mean(ibi_ms), color="red", linestyle="--", label=f"mean={np.mean(ibi_ms):.0f}ms")
        axes[2].set_title("IBI (Inter-Beat Interval)")
        axes[2].set_xlabel("timp (s)")
        axes[2].set_ylabel("IBI (ms)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "Batai insuficiente pentru IBI", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=12, color="red")

    plt.tight_layout()
    out = "ppg_signal.png"
    plt.savefig(out, dpi=150)
    print(f"[OK] Salvat: {out}")


if __name__ == "__main__":
    main()
