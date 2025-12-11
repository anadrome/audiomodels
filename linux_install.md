# Linux Installation

Installation on Linux is fairly straightforward, [unlike
macOS](macos_install.md). But there are a few quirks, so documented here for
completeness.

## Stable Audio Open installation

```bash
python3 -m venv venv_stableaudio
source venv_stableaudio/bin/activate
pip install torch torchaudio diffusers transformers accelerate soundfile huggingface_hub torchsde
```

Note: Requires a Hugging Face token for an account that has accepted the Stable
Audio Open EULA. Put your token in the `HF_TOKEN` environment variable.

## AudioLDM2 installation

1.  Clone the repository:
    ```bash
    mkdir -p repos
    git clone https://github.com/haoheliu/AudioLDM2.git repos/AudioLDM2
    ```

2.  Install:
    ```bash
    python3 -m venv venv_audioldm2
    source venv_audioldm2/bin/activate
    pip install wheel setuptools
    pip install -e repos/AudioLDM2
    ```

## MMAudio installation

MMAudio requires downgrading to a pre-1.x `huggingface_hub`.0.

1.  Clone the repository:
    ```bash
    git clone https://github.com/hkchengrex/MMAudio.git repos/MMAudio
    ```

2.  Install:
    ```bash
    python3 -m venv venv_mmaudio
    source venv_mmaudio/bin/activate
    pip install "huggingface_hub<1.0"
    pip install -e repos/MMAudio
    pip install torchaudio
    ```
