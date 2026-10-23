# EMG Speech — subvocal recognition from a 9-channel ESP32 stream

A complete acquisition, training and real-time recognition project.  It is
built so that the part students need to read is intentionally tiny:

## Start here: students

| Action | Command | What it shows |
|--------|---------|---------------|
| Set the signal shape | `python configure.py` | Writes `emg_config.json` |
| Record examples | `python record.py` | One word per button hold |
| Train a model | `python train.py` | Dataset → checkpoint |
| Recognise live | `python recognize.py` | Checkpoint → text |
| Test without hardware | `python simulate.py --pty` | Fake ESP32 stream |

The only student-facing programs are:

```
configure.py             write the project-wide signal configuration
record.py                record labelled examples (with printed steps)
train.py                 train a classifier (with printed steps)
recognize.py             run live recognition (with printed steps)
simulate.py              create a fake ESP32 stream
inspect_take.py          load and print one saved example
docs/STUDENT_GUIDE.md    the classroom explanation
```

Everything difficult — serial framing, filters, event detection, PyTorch and
the PyQt windows — is inside the **`emg_speech` library**.  Students can use the
project without reading it, then open individual library modules when a lesson
needs the details.

The public API is just:

```python
from emg_speech import ProjectConfig, record, train_model, recognize, simulate
```

These imports work after `pip install -e .` (or any normal installation).  The
entry scripts also run directly from a checkout without installing the package.

> **Students can stop after this section.**  The rest of this README is a
> developer reference; `docs/STUDENT_GUIDE.md` explains the pipeline without
> the implementation details.

Two desktop applications are built on that library:

| App | Entry point | Job |
|-----|-------------|-----|
| **Acquisition & annotation** | `python record.py` | Record one word per button hold, detect the real muscle burst, normalise it to a fixed tensor, save `.npy` per class |
| **Training & inference** | `python train.py` / `python recognize.py` | Train a CNN + BiLSTM classifier, then stream UART, detect gestures and type recognised words live |

The PyQt layer only renders state and forwards user intent. Serial I/O, DSP, file
I/O and PyTorch all live in `src/emg_speech/` and run on worker threads, so the
UI never blocks. `emg_speech` is importable without Qt (and the DSP modules are
importable without torch), which is enforced by `tests/test_architecture.py`.

---

## 1. Architecture

```
              ┌─────────── src/emg_speech/ui/ (PyQt6) ──────────┐
              │  AcquisitionWindow        InferenceWindow       │
              │  widgets / theme          transcript, meter     │
              └───────▲───────────────────────────▲─────────────┘
                      │ Qt signals                │ Qt signals
        ┌─────────────┴─────────────┐   ┌─────────┴──────────────┐
        │ SerialReaderThread        │   │ InferenceThread        │
        │ (QThread: UART read only) │   │ (+ model + recognizer) │
        └─────────────▲─────────────┘   └─────────▲──────────────┘
                      │                           │
              ┌───────┴───────┐           ┌───────┴────────────┐
              │ sources.py    │           │ dsp.py             │
              │ SerialSource  │           │ filters + detector │
              │ ReplaySource  │           │ inference.py       │
              └───────────────┘           └───────▲────────────┘
                                                  │
        ExtractionTask (QThreadPool) ─────────────┤
        dsp.extract_sample -> storage.save_sample │
                                                  │
                                          model.py (torch)
```

Key rule: **one direction only.** UI → worker → core; results come back as
signals. Nothing in `src/emg_speech/ui/` reads a serial port, applies a filter
or touches a tensor.

### Directory layout

```
LICENCE                     MIT license text
README.md                   this file
pyproject.toml              package metadata and build configuration
MANIFEST.in                 extra files for the source distribution
requirements.txt            runtime dependencies for `pip install -r`

record.py                   direct entry script: record examples
train.py                    direct entry script: train a model
recognize.py                direct entry script: live recognition
simulate.py                 direct entry script: fake serial stream
inspect_take.py             view one saved .npy example
_bootstrap.py               lets the scripts import from src/ in a checkout

src/
    emg_speech/             the library — all implementation
        __init__.py         exports record, train_model, recognize, simulate
        api.py              the small public API used by the scripts
        constants.py        channel order, file names, format version
        config.py           DspConfig, DetectionConfig, ModelConfig, TrainConfig,
                            DatasetMeta + UART bandwidth helpers
        protocol.py         defensive UART line -> (9,) float32 frame
        sources.py          SerialSource/Writer, PtyWriter, ReplaySource,
                            port discovery
        ring_buffer.py      lock-free-enough frame ring for take extraction
        dsp.py              causal filters, EventDetector, extract_sample,
                            time normalisation, NoiseProfile
        storage.py          class folders, atomic .npy writes, dataset scan
        dataset.py          DatasetIndex, EMGDataset, NormStats, splits, weights
        model.py            EMGSequenceClassifier, checkpoint save/load
        train.py            training loop, metrics, early stopping
        inference.py        StreamingRecognizer state machine
        synthetic.py        hardware-free dataset generator
        simulator.py        live raw-signal source for virtual serial testing
        ui/                 PyQt6 presentation layer (internal)
        apps/               CLI wrappers used by the scripts (internal)
docs/STUDENT_GUIDE.md       simple explanation for a classroom
docs/PUBLISHING.md          install remotely / publish to PyPI
tests/                      test suite (protocol, DSP, storage, model,
                            training, inference, simulator/PTY, UI, architecture)
```

### UART protocol

One frame per line, either keyed (firmware format) or positional:

```
emg1:512,emg2:498,emg3:505,gx:1.2,gy:-0.4,gz:0.1,ax:0.02,ay:-0.01,az:0.98\n
```

`protocol.LineFramer` tolerates partial reads, oversized junk lines and boot
logs; `FrameParser` counts malformed lines instead of raising. Unknown extra
keys are ignored, so the firmware can add fields without breaking old builds.

**Baud rate matters.** A keyed frame is ~110 bytes, i.e. ~1100 UART bits with
start/stop bits. At 500 Hz that is ~550 kbit/s. The default is therefore
**921600 baud**; `config.baudrate_warning()` flags a too-slow selection in both
apps.

| Frames/s | Minimum practical baud |
|---------:|-----------------------:|
| 250 | 460800 |
| **500** | **921600** |
| 1000 | 1500000 |
| 2000 | 2000000 |

---

## 2. Application 1 — acquisition & annotation

### Workflow

1. Create a class ("Add" → folder `dataset/<word>/`).
2. Select the class, press **Connect**.
3. **Hold** the record button (or Space), say/subvocalise the word, **release**.
4. A take is sliced as `[press − 350 ms pre-roll, release + 250 ms tail]`.
5. DSP runs in a `QThreadPool` task: band-pass → envelope → burst detection →
   crop with margins → resample to `seq_len` → save atomically.
6. The preview shows the three EMG channels by default (**EMG only** is ticked).
   Untick it to show all nine accepted-take channels (EMG and IMU, z-scored per
   channel for readability). Rejected takes display the measured peak vs.
   threshold, so you can fix electrode contact or `on_sigma`.

Settings (`sample rate`, `seq_len`) are written to `dataset/dataset.json` and
locked as soon as one sample exists. Emptying the dataset (delete class /
delete last take) unlocks them again; adding samples at a different rate or
length is refused instead of silently mixing shapes.

For a new project, create the same signal shape once with `emg-config` (or
`python configure.py`) and all commands use it as their default. Existing
`dataset.json` and checkpoint files remain authoritative, so an old dataset or
model can never be silently changed by editing the project config.

### DSP chain (identical offline and live)

```
raw frame
  -> 4th-order Butterworth band-pass 20–120 Hz   (causal, sosfilt with state)
  -> |x|                                          (full-wave rectification)
  -> 2nd-order low-pass 8 Hz                      (envelope)
  -> max over EMG1..EMG3                          (one detection signal)
  -> IMU: 2nd-order low-pass 20 Hz                (kept in physical units)
```

`StreamingPipeline` keeps `sosfilt` state per channel, so pushing frames one by
one gives exactly the same output as `process_array()` over a whole take.

### Event detector

`EventDetector` is a two-threshold hysteresis state machine over the envelope:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `on_sigma` | 4.0 | start when `mean + 4σ` exceeded for 20 ms |
| `off_sigma` | 2.0 | stop when below `mean + 2σ` for 60 ms |
| `min_event_ms` | 150 | shorter bursts are rejected |
| `max_event_ms` | 1600 | longer activity is truncated |
| `pre/post_margin_ms` | 40 / 100 | context kept around the burst |
| `refractory_ms` | 220 | ignore re-triggers after an event |
| `adapt_noise` | on, τ = 4 s | slow noise-floor tracking while idle |
| `calibration_ms` | 2000 | live calibration window |

`NoiseProfile` estimates idle mean/σ with the median and MAD (robust to a
movement spike), falling back to plain σ for short windows. In acquisition each
take is calibrated on its **own pre-roll** (`noise_profile=None`), so a profile
measured minutes earlier cannot mask a gesture after the electrodes settle.
Live inference calibrates once on `calibration_ms` of silence.

### Normalisation

* **Time:** linear resampling to exactly `seq_len` (default 128) — uniform
  tensor shapes without losing the event.
* **EMG amplitude:** each sample's EMG channels are scaled to unit RMS
  (`rms_floor=1e-3`), so the model learns shape, not effort.
* **IMU:** left in physical units and standardised with train-set statistics
  stored inside the checkpoint.

Saved arrays are `float32`, shape `(128, 9)`, channel order fixed by
`constants.CHANNELS`. Writes are atomic (`tmp` + `os.replace`).

---

## 3. Application 2 — training & real-time inference

### Model

```
(x: B, 128, 9)
  -> Conv1d stem (64 ch, kernel 5, same padding, GELU, dropout 0.1)
  -> 2 × BiLSTM(96) with dropout
  -> additive attention pooling over time
  -> LayerNorm -> Linear(2·96, n_classes)
```

Normalisation buffers (`mean`, `std`) are registered in the checkpoint, so the
model receives standardised input without the UI having to know about it.

### Training

`DatasetIndex.build()` validates every file, drops unreadable/wrong-shape
samples with warnings, and **remaps labels** when a class has no valid samples
left (that class is reported and excluded). `EMGDataset` is a standard
`torch.utils.data.Dataset` returning `(x, y)`; `stratified_split()` guarantees
every class appears in train *and* validation.

The loop uses `CrossEntropyLoss` (inverse-frequency class weights by default),
`Adam`, gradient clipping, `ReduceLROnPlateau` and early stopping. The best
checkpoint embeds:

* model `state_dict` + `ModelConfig`
* label map
* `DspConfig` and `DetectionConfig`
* normalisation statistics
* metrics (accuracy, macro-F1, confusion matrix, class counts)

so inference can never drift from training.

### Streaming inference

`StreamingRecognizer` runs a single state machine inside `InferenceThread`:

```
CALIBRATING --(2 s of idle)--> LISTENING
LISTENING   --(onset debounce)--> ACTIVE   (keeps a bounded frame deque)
ACTIVE      --(offset debounce + post margin)--> predict -> LISTENING
```

Only the current event is buffered (`pre + max_event + post` frames), so memory
is constant. The first forward pass is warmed up at load time; steady-state
model latency is ~10–15 ms on CPU for the default architecture. Predictions
below the confidence threshold are **shown but not typed**, and a
`Recalibrate` button restarts the noise-floor calibration without reconnecting.

`ReplaySource` feeds a recorded dataset through the exact same path, which makes
the whole UI demoable without hardware.

---

## 4. PyQt UI

Both windows are `QMainWindow` subclasses that own only widgets and a worker
reference. They never call `read()`, `sosfilt()` or `model()` directly.

* **Acquisition** — class list, 30 fps scrolling envelope with threshold line,
  live level meter, normalised take preview, detection settings, hold-to-record
  button (mouse or Space).
* **Inference** — checkpoint picker with metrics, confidence slider, calibration
  progress, last-prediction detail, and a large transcript with Clear / Copy /
  Save. The **Connect/Disconnect** button in the port selector and the **Start
  listening** button control the same stream. The live view is a fixed-window
  strip chart (it scrolls instead of shrinking the whole session): the default
  view is the detection envelope, and **All 9 signals** shows every filtered
  channel z-scored per channel. The envelope threshold stays visible in both
  views.

Threading contract:

```python
thread = SerialReaderThread(lambda: SerialSource(port, baud), parent=self)
thread.frames_batch.connect(self._on_frames)
thread.opened.connect(self._on_serial_opened)
thread.error.connect(self._on_error)
thread.closed.connect(self._on_serial_closed)
thread.finished.connect(self._on_thread_finished)
thread.start()
```

* UART reads and DSP happen off the GUI thread; frames arrive in ≤20 ms batches.
* `ExtractionTask` (a `QRunnable`) does DSP + disk I/O for a take.
* Link loss is retried with backoff; parse errors are counted, not fatal.
* Shutdown is idempotent: `closed` and `finished` both reset the controls, the
  pool is drained on close, and a blocked driver read is terminated after 4 s.

---

## 5. Installation & usage

### Dependencies

```bash
python -m venv .venv && source .venv/bin/activate

# Recommended: install this project in editable mode.
python -m pip install -e ".[dev]"

# Or install only the runtime dependencies.  The entry scripts still run from
# a checkout because _bootstrap.py adds src/ to sys.path, but the installed
# console commands and `python -m emg_speech...` need the editable install.
# pip install -r requirements.txt
```

Requires Python ≥ 3.10, `numpy`, `scipy`, `pyserial`, `torch`, `PyQt6`
(`pytest` for the test suite). CUDA/XPU/MPS is used automatically when
available (`--device auto`).  The package uses the standard `src/` layout from
the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

### Install on another computer

The project is a normal Python package.  After pushing it to a Git repository:

```bash
pip install "git+https://github.com/<your-user>/<your-repo>.git"
```

Or, after publishing to PyPI:

```bash
pip install emg-speech
```

This installs the library and these console commands:

```bash
emg-config --help
emg-record --help
emg-train --help
emg-recognize --help
emg-simulate --help
```

`docs/PUBLISHING.md` contains the full build, TestPyPI, PyPI and API-token
instructions.  Datasets and `artifacts/` are **not** packaged; create them on
the target machine using `emg-record` / `emg-train`.

### Project-wide configuration

Create `emg_config.json` once in the project folder.  The installed command and
the direct script do exactly the same thing:

```bash
emg-config --sample-rate 1000 --seq-len 256
# or, from a checkout:
python configure.py --sample-rate 1000 --seq-len 256
```

The file records the signal shape and detector settings:

```json
{
  "format_version": 1,
  "dsp": {"sample_rate": 1000.0, "seq_len": 256, "...": "..."},
  "detection": {"on_sigma": 4.0, "...": "..."}
}
```

Use `emg-config` with no options to show the current values, `--config PATH` to
keep several settings files, and `--reset` to restore the defaults.  The
recording window, `simulate.py`, training and inference all read the file
automatically; each command also accepts `--config PATH`.

Precedence is intentional:

- A **new** dataset uses the project config for its initial sample rate,
  sequence length and detector settings.
- Once `dataset/dataset.json` exists with samples, that file is authoritative.
  Training reports a clear error if the project config disagrees.
- A **checkpoint** stores the DSP settings it was trained with.  Inference
  reports a clear error if the project config disagrees, instead of running a
  model on the wrong signal shape.
- The simulator uses the project sample rate unless `--rate` is given
  explicitly.

### No-hardware demo

```bash
# 1. 4 classes x 20 synthetic takes through the real DSP pipeline
emg-demo-data --out dataset_demo

# 2. train (CPU is fine)
python train.py --data dataset_demo --out artifacts/model.pt --epochs 20

# 3. replay the dataset through the live UI
python recognize.py --checkpoint artifacts/model.pt --replay dataset_demo
```

### With hardware

```bash
python configure.py --sample-rate 500 --seq-len 128
python record.py --data dataset --port /dev/ttyUSB0 --baud 921600
python train.py   --data dataset --out artifacts/model.pt --epochs 60
python recognize.py   --checkpoint artifacts/model.pt --port /dev/ttyUSB0 --baud 921600
```

Use `--port COM5` on Windows. The firmware must sample and transmit at the same
rate stored in `emg_config.json` (or the dataset/checkpoint for existing work)
and use the same baud rate.

#### Windows COM ports

The port box is **editable** in both windows: pick a detected port or click it
and type one directly (`COM3`, `COM10`, …). Ports that are not enumerated —
including a com0com pair or a POSIX PTY — still work. The refresh button keeps
your typed name, so a temporarily unplugged adapter does not clear the field.
On the command line the same applies:

```bash
python record.py --data dataset --port COM3 --baud 921600
python recognize.py --checkpoint artifacts/model.pt --port COM3
```

#### Virtual serial testing (no ESP32)

The simulator generates a continuous **raw** 9-channel stream (idle noise +
class-specific bursts and motion) and writes it to a serial endpoint. It does
not replay normalised dataset samples: the acquisition window, the threaded
serial reader, the DSP/detector and the inference state machine all run exactly
as they do with hardware.

On Linux/macOS, `--pty` creates a virtual pair with no extra dependencies:

```bash
# terminal 1 — note the printed path, e.g. /dev/pts/7
python simulate.py --pty

# terminal 2 — type /dev/pts/7 into the port box, or:
python record.py --data dataset --port /dev/pts/7
```

`auto` mode emits a gesture every `--interval` seconds and prints a cue; press
and hold record around the cue. `manual` mode is best for annotation practice —
type a word and press Enter while the record button is held:

```bash
python simulate.py --pty --mode manual --words hello,world
# [sim] 1.23s gesture 'hello'
```

Inference over the fake UART (the bundled demo checkpoint understands the
default vocabulary):

```bash
python simulate.py --pty --interval 1.5
python recognize.py --checkpoint artifacts/model.pt --port /dev/pts/7
```

Useful flags: `--noise`, `--amplitude`, `--start-idle`, `--seed`,
`--list-ports`, and `--dry-run --duration 6` (unpaced self-test that reports
cues, encoded frame size and whether the detector finds the burst).

On **Windows**, create a null-modem pair once with
[com0com](https://sourceforge.net/projects/com0com/) from an elevated prompt:

```bat
setupc.exe install PortName=COM5 PortName=COM6
python simulate.py --setup-windows     :: prints the same recipe
```

Then run the simulator on one end and the app on the **other** end (Windows
does not let two processes open the same COM port):

```bat
python simulate.py --port COM5 --baud 921600
python record.py --port COM6 --baud 921600
python recognize.py   --checkpoint artifacts\model.pt --port COM6
```

A virtual pair ignores the physical baud rate, but keep the same value on both
sides. `--replay dataset_demo` streams recorded dataset samples instead of the
synthetic generator; the apps' own `--replay` mode remains the fastest way to
test inference without any serial layer.

### Minimal ESP32 firmware shape

```cpp
#include <Wire.h>
#include <MPU6050.h>

MPU6050 imu;
const uint32_t PERIOD_US = 2000;          // 500 Hz
uint32_t next_us = 0;

void setup() {
  Serial.begin(921600);
  Wire.begin();
  imu.initialize();
  next_us = micros();
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - next_us) < 0) return;

  int16_t ax, ay, az, gx, gy, gz;
  imu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  int e1 = analogRead(34), e2 = analogRead(35), e3 = analogRead(32);

  char line[128];
  int n = snprintf(line, sizeof(line),
      "emg1:%d,emg2:%d,emg3:%d,gx:%.2f,gy:%.2f,gz:%.2f,"
      "ax:%.3f,ay:%.3f,az:%.3f\n",
      e1, e2, e3,
      gx / 131.0f, gy / 131.0f, gz / 131.0f,
      ax / 16384.0f, ay / 16384.0f, az / 16384.0f);
  Serial.write((const uint8_t*)line, n);

  next_us += PERIOD_US;
  if ((int32_t)(now - next_us) > (int32_t)(PERIOD_US * 4)) next_us = now + PERIOD_US;
}
```

Notes:

* A 12-bit ADC centred near 2048 is fine; the detector adapts to any DC offset.
* Keep EMG electrodes dry and far from the mains; 20–120 Hz rejects most motion
  and mains hum, but a notch at 50/60 Hz can be added in `DspConfig` if needed.
* If `Serial.write` occasionally blocks, lower the rate or raise the baud —
  dropped frames look like jittery envelopes in the UI.

### Tuning

| Symptom | Fix |
|---------|-----|
| Nothing detected, peak far below threshold | Lower `on_sigma` (3.0), check electrodes, raise gain |
| Noise triggers takes | Raise `on_sigma`, improve grounding, keep still during calibration |
| Words cut at the end | Increase `post_margin_ms` or hold slightly longer before releasing |
| Two bursts kept per take | The strongest is saved; the UI warns. Slow down between repetitions |
| Confidence always low | More takes per class (20–40), consistent pronunciation, enable `--augment` |
| Frames arrive late / truncated | Use ≥921600 baud or lower the sample rate |

### Tests

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=. python -m pytest -q
```

83 tests cover protocol edge cases, filter/detector behaviour, normalisation,
atomic storage, dataset remapping, model/checkpoint round-trips, training,
streaming inference, the live simulator (including a POSIX PTY round-trip),
the public library API, the four directly runnable entry scripts, layer
boundaries and offscreen UI tests for manual COM-port entry, the 3/9-channel
take preview, inference connect wiring, and the scrolling all-signal view.

---

## Dataset on disk

```
dataset/
    dataset.json                     rate, seq_len, channel order, DSP config
    hello/
        hello_20261022-101500_ab12cd34.npy     (128, 9) float32
    world/
        ...
```

Only `.npy` files are read; empty folders never reach training.
