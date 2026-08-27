# remote-dpd

`remote-dpd` is the MATLAB-free replacement for the remote DPD file-watch
service. It keeps the existing `.mat` exchange contract and currently exposes
one algorithm: iterative learning control (ILC) DPD. The numerical core uses
NumPy/SciPy and PyTorch tensors; no MATLAB Engine, MATLAB Runtime, or MATLAB
license is required.

## Run

Install the project in a Python environment with PyTorch. On CPU-only
deployment hosts, use the supplied requirements file so pip does not pull
CUDA runtime packages:

```bash
python -m pip install -r requirements-cpu.txt
python -m pip install -e . --no-deps
remote-dpd Zilink --watch-root /opt/SharePoint
```

The service watches `/opt/SharePoint/<SUPPLIER_NAME>` by default. A custom
directory can be supplied with `--path`; this is useful for a staging folder.
The process is intentionally resident until interrupted.

## Exchange contract

The service accepts the existing files `Config_file.mat`, `DPD_in.mat`, and
`FB_Signal.mat` and writes `Config_file_ack.mat`, `ACK_DPDin.mat`,
`DPDout_Nokia.mat`, `symbolEVM.mat`, and the periodic `sync_dat.txt` heartbeat.
MAT v5/v6 files are handled with `scipy.io`; MATLAB v7.3/HDF5 files are
accepted when `h5py` is installed.

The legacy `configDPD` struct is accepted. Its `run_idealDPD`, `enILC`, and
`idealDPD` flags are treated as compatibility metadata: this service always
runs the supported ILC engine and never selects the old MARS/MADE engines.

## Architecture

* `protocol.py` contains the file names and MAT conversion at the boundary.
* `config.py` converts legacy MATLAB structs to a typed ILC configuration.
* `dsp.py` contains alignment, circular FIR, optional resampling, and metrics.
* `algorithms.py` defines the extensible engine interface and the ILC engine.
* `service.py` owns the explicit session state and file-watch protocol.

The existing file watcher can still be extended programmatically through
`DPDEngine.process(...)`. New device-controlled development uses the versioned
`DPDRuntime` contract below; the legacy engine interface will leave with the
legacy watcher in a later refactoring stage.

The first refactoring stage also provides device-independent contracts that
are not yet wired into the file-watch entry point:

* `device.py` defines typed RF device capabilities and adapter schemas.
* `preprocessing.py` aligns and coherently averages capture batches while
  keeping the first-round gain correction fixed.
* `runtime.py` defines the versioned DPD runtime contract and basic ILC.
* `safety.py` enforces the fixed digital peak and RMS limits without AGC,
  scaling, or clipping.

See [`docs/README.md`](docs/README.md) for the current design and refactoring
status. Simulation, orchestration, the new file protocol, and the web console
are intentionally deferred to later stages.

## Important compatibility note

The old MATLAB implementation stores state in the MATLAB base workspace. The
Python implementation stores it explicitly in `SessionState`, which is reset
by `Config_file.Reset` or when a new `DPD_in` waveform arrives. This makes
restart and testing deterministic while preserving the observable file API.
