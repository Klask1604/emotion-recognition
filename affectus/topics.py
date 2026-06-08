"""
Sursa UNICA de adevar pentru topicurile MQTT de productie ale sistemului.

Schema e organizata pe DIRECTIA fluxului, ca fiecare topic sa-si spuna rolul din nume:
  - In.*   : intra in motor (de la ceas sau Unity)
  - Out.*  : ies din motor (catre ceas / VR / Grafana)

Topicurile de cercetare (`biofizic/legacy/*`, `biofizic/test/*`) NU sunt aici: ele
raman pe schema veche, sunt efemere si nu fac parte din protocolul de productie.

Numele de measurement InfluxDB sunt DECUPLATE de numele topicului (vezi
services/mqtt_logger.py): topicul se poate redenumi fara a atinge dashboardurile
Grafana, care interogheaza measurement-urile dupa numele lor istoric.
"""

from __future__ import annotations

PREFIX = "biofizic"


class In:
    """Topicuri de intrare in motor (ceas -> engine, Unity -> engine)."""

    ACQUISITION = f"{PREFIX}/in/acquisition"   # cadre senzori + PPG 100 Hz inclus, 1 Hz
    HELLO = f"{PREFIX}/in/hello"               # handshake: ceasul isi anunta capabilitatile
    CMD_CALIBRATE = f"{PREFIX}/in/cmd/calibrate"  # comanda de recalibrare baseline
    CMD_FEEDBACK = f"{PREFIX}/in/cmd/feedback"    # eticheta de emotie de la buton
    CONTEXT = f"{PREFIX}/in/context"           # context scena (Unity -> engine)


class Out:
    """Topicuri de iesire din motor (engine -> ceas / VR / Grafana)."""

    HELLO_ACK = f"{PREFIX}/out/hello/ack"      # raspuns handshake (ok/eroare + capabilitati acceptate)
    AROUSAL = f"{PREFIX}/out/arousal"          # verdict de epoca (activare), retinut pt bootstrap
    LIVE = f"{PREFIX}/out/live"                # activare neteda 1 Hz, pt UI ceas
    WINDOWS = f"{PREFIX}/out/windows"          # ferestre HRV w30/w60/w90, pt Grafana
    EMOTION = f"{PREFIX}/out/emotion"          # cadran Russell + emotii tipice (ceas + Grafana)
    CALIBRATION = f"{PREFIX}/out/calibration"  # status calibrare (faza: collecting/done)
