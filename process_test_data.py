"""
Battery pack vibration test - data processing and report script
----------------------------------------------------------------
Processes one log from the vibration test described in the SOP:
  * accelerometer data (control on fixture plate, response on pack lid)
  * thermocouple log
  * BMS messages on CAN, decoded against a DBC definition

IMPORTANT: the input data here is SYNTHETIC, generated in this script so the
whole processing chain can be demonstrated without hardware. To use real logs,
replace load_synthetic_log() with a loader for your DAQ/CAN files; the analysis
and report code below does not change.

Author: Kailash Srungavarapu
"""
import numpy as np
import cantools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

rng = np.random.default_rng(7)

# ---------------- test definition (edit to match the applicable standard) ----------
F_START, F_PEAK = 7.0, 50.0          # Hz, log sine sweep 7 -> 50 -> 7 Hz (R100-style)
SWEEP_UP_S = 450.0                   # s, one half of a 15 min cycle
FS = 1000.0                          # Hz, accelerometer sample rate
FIXTURE_F1_HZ = 786.0                # Hz, ANSYS modal, 60 kg pack as point mass, plate fixed at the 24 bolt holes
TEST_UPPER_HZ = 200.0                # Hz, conservative band edge used for fixture sizing
FIXTURE_MARGIN_MIN = 1.5             # fixture f1 / upper test frequency, minimum accepted

# ---------------- CAN definition (BMS status frame) ----------------
DBC = """
VERSION ""
NS_ :
BS_:
BU_: BMS LOGGER
BO_ 1008 BMS_Status: 8 BMS
 SG_ PackVoltage : 0|16@1+ (0.1,0) [0|500] "V" LOGGER
 SG_ PackCurrent : 16|16@1- (0.1,0) [-300|300] "A" LOGGER
 SG_ CellTempMin : 32|8@1+ (1,-40) [-40|125] "degC" LOGGER
 SG_ CellTempMax : 40|8@1+ (1,-40) [-40|125] "degC" LOGGER
"""


def load_synthetic_log():
    n = int(SWEEP_UP_S * FS)
    t = np.arange(n) / FS
    k = np.log(F_PEAK / F_START) / SWEEP_UP_S
    phase = 2 * np.pi * F_START * (np.exp(k * t) - 1) / k      # logarithmic sweep
    control = 1.0 * np.sin(phase) + 0.01 * rng.standard_normal(n)

    # pack lid modelled as a single-degree-of-freedom system, fn = 33 Hz, zeta = 0.06
    fn, zeta = 33.0, 0.06
    wn = 2 * np.pi * fn
    b, a = [2 * zeta * wn, wn**2], [1, 2 * zeta * wn, wn**2]
    _, resp, _ = signal.lsim((b, a), control, t)
    resp = resp + 0.01 * rng.standard_normal(n)

    # thermocouples: 8 channels, 1 Hz, slow rise from ambient, one runs warmer
    tt = np.arange(0, SWEEP_UP_S, 1.0)
    tc = 23.0 + 0.8 * (1 - np.exp(-tt / 200.0))[:, None] * np.linspace(0.8, 1.2, 8)[None, :]
    tc += 0.05 * rng.standard_normal(tc.shape)

    # CAN: encode then decode so the DBC round trip is exercised, 10 Hz
    db = cantools.database.load_string(DBC, "dbc")
    msg = db.get_message_by_name("BMS_Status")
    frames = []
    for ts in np.arange(0, SWEEP_UP_S, 0.1):
        data = msg.encode({
            "PackVoltage": 355.0 + 0.2 * rng.standard_normal(),
            "PackCurrent": 0.0 + 0.1 * rng.standard_normal(),
            "CellTempMin": 22.0 + 0.8 * (1 - np.exp(-ts / 200.0)),
            "CellTempMax": 23.0 + 1.0 * (1 - np.exp(-ts / 200.0)),
        })
        frames.append((ts, msg.frame_id, data))
    return t, control, resp, tt, tc, db, frames


def decode_can(db, frames):
    out = {k: [] for k in ("t", "PackVoltage", "PackCurrent", "CellTempMin", "CellTempMax")}
    for ts, fid, data in frames:
        sig = db.decode_message(fid, data)
        out["t"].append(ts)
        for k in sig:
            out[k].append(sig[k])
    return {k: np.array(v) for k, v in out.items()}


def transmissibility(control, resp):
    f, Pxx = signal.welch(control, FS, nperseg=8192)
    _, Pxy = signal.csd(control, resp, FS, nperseg=8192)
    H = Pxy / Pxx
    return f, np.abs(H)


def main():
    t, control, resp, tt, tc, db, frames = load_synthetic_log()
    can = decode_can(db, frames)
    f, T = transmissibility(control, resp)

    band = (f >= F_START) & (f <= F_PEAK)
    i_pk = np.argmax(np.where(band, T, 0))
    f_res, Q = f[i_pk], T[i_pk]

    margin = FIXTURE_F1_HZ / TEST_UPPER_HZ
    dT = tc[-1].max() - tc[0].min()
    v_drift = can["PackVoltage"].max() - can["PackVoltage"].min()

    summary = [
        f"Pack lid resonance: {f_res:.1f} Hz, amplification {Q:.1f}x",
        f"Fixture f1 / {TEST_UPPER_HZ:.0f} Hz band edge = {margin:.2f}  "
        f"({'OK' if margin >= FIXTURE_MARGIN_MIN else 'BELOW CRITERION'}, min {FIXTURE_MARGIN_MIN})",
        f"Max thermocouple rise during test: {dT:.2f} K",
        f"BMS pack voltage spread (decoded from CAN): {v_drift:.2f} V",
    ]
    print("\n".join(summary))

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(f, T, color="#2E5C8A")
    ax[0, 0].axvline(f_res, color="#C0722C", ls="--")
    ax[0, 0].set_xlim(F_START, F_PEAK); ax[0, 0].set_xlabel("Frequency (Hz)")
    ax[0, 0].set_ylabel("Transmissibility |lid / control|")
    ax[0, 0].set_title(f"Resonance at {f_res:.1f} Hz, Q = {Q:.1f}")

    ax[0, 1].plot(t[::10], resp[::10], lw=0.4, label="response (pack lid)", color="#C0722C", alpha=0.7)
    ax[0, 1].plot(t[::10], control[::10], lw=0.5, label="control (plate)", color="#151D24", zorder=3)
    ax[0, 1].set_xlabel("Time (s)"); ax[0, 1].set_ylabel("Acceleration (g)")
    ax[0, 1].legend(loc="upper left"); ax[0, 1].set_title("Sine sweep, 7 to 50 Hz")

    ax[1, 0].plot(tt, tc, lw=0.8)
    ax[1, 0].set_xlabel("Time (s)"); ax[1, 0].set_ylabel("Temperature (degC)")
    ax[1, 0].set_title("Thermocouples T1-T8")

    ax[1, 1].plot(can["t"], can["CellTempMax"], label="BMS max cell temp", color="#C0722C")
    ax[1, 1].plot(can["t"], can["CellTempMin"], label="BMS min cell temp", color="#2E5C8A")
    ax[1, 1].set_xlabel("Time (s)"); ax[1, 1].set_ylabel("Temperature (degC)")
    ax[1, 1].legend(); ax[1, 1].set_title("CAN-decoded BMS temperatures (DBC, 1 degC resolution)")

    fig.suptitle("Battery pack vibration test - processed report (synthetic data)", y=0.995)
    fig.tight_layout()
    fig.savefig("test_report.png", dpi=150, facecolor="white")
    print("saved test_report.png")


if __name__ == "__main__":
    main()
