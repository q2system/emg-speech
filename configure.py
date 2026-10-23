"""Configure the signal shape used by the whole project.

Steps:
    1. Choose the sample rate that matches your ESP32 firmware.
    2. Choose the number of time steps every recorded take is resampled to.
    3. The command writes emg_config.json in the current folder.
    4. record.py, simulate.py, train.py and recognize.py use it automatically.

Run:
    python configure.py --sample-rate 1000 --seq-len 256
    python configure.py --sample-rate 500 --seq-len 128 --on-sigma 4.5
    python configure.py                 # show the current values
    python configure.py --reset         # restore the built-in defaults
"""

from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

from emg_speech.apps.configure import main


if __name__ == "__main__":
    raise SystemExit(main())
