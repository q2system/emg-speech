# Student guide

This project turns nine sensor channels into recognised words.  The visible
code is deliberately small; the difficult parts are inside the `emg_speech`
library.

## The whole project in one picture

```
ESP32 sensors
    │  9 numbers per frame
    ▼
UART / serial  ──►  filters + envelope  ──►  burst detector
                                                   │
                                                   ▼
                                        one 128 × 9 example
                                                   │
                                                   ▼
                                      CNN + BiLSTM classifier
                                                   │
                                                   ▼
                                              word / text
```

One saved example has shape **`(128, 9)`**:

- **128** time steps, after the detected burst is resampled to a fixed length.
- **9** channels: `emg1`, `emg2`, `emg3`, `gx`, `gy`, `gz`, `ax`, `ay`, `az`.

## The three core operations

The Python snippets below assume the package is installed
(`pip install -e .`).  The four scripts run directly from a checkout.

### 1. Record

```python
from emg_speech import record

record("dataset")
```

This opens the acquisition window.  Hold the record button, say the word, and
release.  The library detects the muscle burst and saves a normalised example
under `dataset/<word>/`.

### 2. Train

```python
from emg_speech import train_model

report = train_model("dataset", "artifacts/model.pt")
print(report.summary())
```

This reads every saved example, splits it into training and validation sets,
trains the neural network, and writes one checkpoint file.  The checkpoint
contains the model weights, the label names, and the DSP settings needed by
recognition.

### 3. Recognise

```python
from emg_speech import recognize

recognize("artifacts/model.pt")
```

This opens the live window.  The library reads the serial stream, detects a
burst, classifies it, and adds the predicted word to the transcript.

## Run the four scripts directly

```bash
python record.py       # 1. record labelled examples
python train.py        # 2. train a model
python recognize.py    # 3. recognise words live
python simulate.py --pty  # optional: fake ESP32 stream
```

Use `python simulate.py --pty` to create a fake ESP32 serial port for
testing without hardware.  The port is printed in the terminal and can be
typed into the app.

Every script prints its steps before doing the work.  For example,
`record.py` prints the dataset folder, serial settings, and when the
acquisition window is opening.

## What is where

Only these top-level files are meant to be read by students:

| Path | Purpose |
|------|---------|
| `record.py` | Open the acquisition window. |
| `train.py` | Train and save a model. |
| `recognize.py` | Open the live recognition window. |
| `simulate.py` | Create a fake ESP32 stream. |
| `inspect_take.py` | Print one saved example. |
| `docs/STUDENT_GUIDE.md` | This explanation. |

Everything else is the library.  It uses the standard `src/` layout, so the
package lives under `src/emg_speech/`; `_bootstrap.py` only helps the four
scripts find it when you run them directly from a checkout.

| Library folder | Responsibility |
|----------------|----------------|
| `src/emg_speech/protocol.py` | Turn UART text lines into 9-channel frames. |
| `src/emg_speech/sources.py` | Read a serial port or replay a dataset. |
| `src/emg_speech/dsp.py` | Filters, envelope, burst detection, resampling. |
| `src/emg_speech/storage.py` | Save and load `.npy` examples safely. |
| `src/emg_speech/dataset.py` | Build train/validation sets and normalise inputs. |
| `src/emg_speech/model.py` | CNN + BiLSTM architecture and checkpoints. |
| `src/emg_speech/train.py` | Training loop and metrics. |
| `src/emg_speech/inference.py` | Real-time state machine. |
| `src/emg_speech/ui/` | PyQt windows, widgets and worker threads. |
| `src/emg_speech/apps/` | Command-line wrappers used by the four scripts. |

The UI never performs signal processing itself.  It sends data to worker
threads and displays the results that come back.  That is why the windows stay
responsive while the model is working.

## Four questions to ask in class

1. **What is a frame?**  Nine numbers from the sensors at one moment.
2. **Why filter?**  To remove slow movement and high-frequency noise, leaving
   the muscle activity that carries speech information.
3. **Why detect a burst?**  To find the part of the recording that contains the
   word instead of saving silence.
4. **Why 128 samples?**  The classifier needs the same input shape every time,
   even if a word is spoken slightly faster or slower.

## One important setting

The sample rate in the dataset configuration must match the firmware.  At
**500 Hz**, there are **2 ms** between frames.  A firmware that sends one frame
every 10 ms is running at **100 Hz**, and the filters, burst timing, and model
window will not match a 500 Hz checkpoint.
