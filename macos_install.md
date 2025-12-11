# macOS Installation

The three audio models we're using don't install and run as-is on macOS, but it
is possible to make them work. This is here so that I remember the installation
process and workarounds.

Tested on macOS Tahoe 26.1, Apple Silicon, Python 3.14, in December 2025 by
Mark.

Note that some of these changes might make the generated audio not identical to
what the upstream repositories produce. Therefore I have only been using this
for local experimentation; the data for papers is generated on Linux by the
unmodified upstream versions.

## Stable Audio Open installation

Stable Audio Open actually installs and loads fine with the PyPI version of
`diffusers`. However the default solver hits a recursion limit on macOS, so the
scripts switch to `DPMSolverMultistepScheduler` on macOS.

```bash
python3 -m venv venv_stableaudio
source venv_stableaudio/bin/activate
pip install torch torchaudio diffusers transformers accelerate soundfile huggingface_hub torchsde
```

Note: Requires a Hugging Face token for an account that has accepted the Stable
Audio Open EULA. Put your token in the `HF_TOKEN` environment variable.

## AudioLDM2 installation

The version of `transformers` that AudioLDM2 depends on doesn't install on
Apple Silicon, due to Rust compilation errors in `tokenizers`. However,
AudioLDM2 can use a newer `transformers` with a small patch.

1.  Clone the repository:
    ```bash
    mkdir -p repos
    git clone https://github.com/haoheliu/AudioLDM2.git repos/AudioLDM2
    ```

2.  Relax version constraints:
    Edit `repos/AudioLDM2/setup.py` as follows:
    - Change `numpy<=1.23.5` to `numpy`
    - Change `librosa==0.9.2` to `librosa`
    - Change `transformers==4.30.2` to `transformers`

3.  Install:
    ```bash
    python3 -m venv venv_audioldm2
    source venv_audioldm2/bin/activate
    pip install wheel setuptools
    pip install -e repos/AudioLDM2
    ```

4.  Patch the code:
    Patch `audioldm2/pipeline.py` to allow unexpected keys in the state
    dictionary (due to the `transformers` API changing).
    
    In `repos/AudioLDM2/audioldm2/pipeline.py`, change:
    ```python
    latent_diffusion.load_state_dict(checkpoint["state_dict"])
    ```
    to:
    ```python
    latent_diffusion.load_state_dict(checkpoint["state_dict"], strict=False)
    ```

## MMAudio installation

MMAudio requires relaxing a few dependency relaxations and downgrading to a
pre-1.x `huggingface_hub`.

1.  Clone the repository:
    ```bash
    git clone https://github.com/hkchengrex/MMAudio.git repos/MMAudio
    ```

2.  Relax version constraints:
    Edit `repos/MMAudio/pyproject.toml` as follows:
    - Change `numpy >= 1.21, <2.1` to `numpy >= 1.21`
    - Change `tensordict >= 0.6.1` to `tensordict`

3.  Install:
    ```bash
    python3 -m venv venv_mmaudio
    source venv_mmaudio/bin/activate
    pip install "huggingface_hub<1.0"
    pip install -e repos/MMAudio
    pip install torchaudio
    ```
