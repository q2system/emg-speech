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
(`pip install -e .`).  The entry scripts run directly from a checkout.

### 1. Record

```python
from emg_speech import record

record("dataset")
```

This opens the acquisition window.  Hold the record button, say the word, and
release.  The library detects the muscle burst and saves a normalised example
under `dataset/<word>/`.

Double-click a class name to link an optional sentence to it.  The sentence is
stored in `dataset/dataset.json` and copied into the checkpoint during training.

### 2. Train

```python
from emg_speech import train_model

report = train_model("dataset", "artifacts/model.pt")
print(report.summary())
```

This reads every saved example, splits it into training and validation sets,
trains the neural network, and writes one checkpoint file.  The checkpoint
contains the model weights, the label names, any linked class sentences, and the
DSP settings needed by recognition.

When training finishes, `report.summary()` shows these scores for both the
training and validation sets:

- accuracy and error rate
- sensitivity (recall)
- specificity
- precision
- F1 score

Sensitivity, specificity, precision and F1 are averaged across the word classes,
so the same table works whether you have two words or ten.

### 3. Recognise

```python
from emg_speech import recognize

recognize("artifacts/model.pt")
```

This opens the live window.  The library reads the serial stream, detects a
burst, classifies it, and adds the predicted word to the transcript.  If the
class has a linked sentence, the transcript shows that sentence instead.  The
**Speak words** checkbox also reads the displayed word or sentence aloud through
the system text-to-speech engine.  It is off by default and is disabled when the
computer has no usable speech engine.

## See every step with plots

`pipeline_walkthrough.py` is a single classroom file that starts from the
examples already saved in `dataset/`.  It walks through preprocessing,
normalisation, batches, the model architecture, a forward pass, loss, the
training loop, and the final confusion matrix.  Every section opens one figure
and prints a short explanation.

```bash
python pipeline_walkthrough.py --data dataset_demo --epochs 3
```

Use `--save-dir plots` to write eight PNG files instead of opening windows.
The main file is deliberately short and only calls one helper per section.  All
Matplotlib setup and the longer functions are in `helper_lib/`, which keeps the
pipeline easy to follow.  The walkthrough trains a small model for teaching;
`train.py` is still the command that produces the real checkpoint.

## Run the scripts directly

```bash
python configure.py --sample-rate 500 --seq-len 128  # 0. set the signal shape
python record.py       # 1. record labelled examples
python train.py        # 2. train a model
python recognize.py    # 3. recognise words live
python simulate.py --pty  # optional: fake ESP32 stream
python pipeline_walkthrough.py --data dataset_demo  # plotted tour
```

`configure.py` writes `emg_config.json` in the current folder.  The other
scripts read it automatically, so the sample rate and sequence length only
need to be typed once.  The installed command `emg-config` does the same thing.

```bash
emg-config --sample-rate 1000 --seq-len 256
emg-config                # show the current values
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
| `configure.py` | Write `emg_config.json` for the whole project. |
| `record.py` | Open the acquisition window. |
| `train.py` | Train and save a model. |
| `recognize.py` | Open the live recognition window. |
| `simulate.py` | Create a fake ESP32 stream. |
| `inspect_take.py` | Print one saved example. |
| `pipeline_walkthrough.py` | Plot the whole pipeline from saved data to training results. |
| `helper_lib/` | Plot setup and the longer helper functions used by the walkthrough. |
| `docs/STUDENT_GUIDE.md` | This explanation. |

Everything else is the library.  It uses the standard `src/` layout, so the
package lives under `src/emg_speech/`; `_bootstrap.py` only helps the scripts
find it when you run them directly from a checkout.

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
| `src/emg_speech/apps/` | Command-line wrappers used by the entry scripts. |

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

The sample rate must match the firmware.  At **500 Hz**, there are **2 ms**
between frames.  A firmware that sends one frame every 10 ms is running at
**100 Hz**, and the filters, burst timing, and model window will not match a
500 Hz checkpoint.

Set it once with `python configure.py --sample-rate 500 --seq-len 128` (or
`emg-config`).  New datasets copy those values into `dataset/dataset.json`.
That file and the trained checkpoint are authoritative after data exists, so
the tools refuse to train or run inference with a config that disagrees with
them.
