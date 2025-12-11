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

```bash
python3 -m venv venv_audioldm2
source venv_audioldm2/bin/activate
pip install wheel setuptools
pip install git+https://github.com/haoheliu/AudioLDM2.git
```

## MMAudio installation

MMAudio requires downgrading to a pre-1.x `huggingface_hub` and is missing a
dependency.

```bash
python3 -m venv venv_mmaudio
source venv_mmaudio/bin/activate
pip install "huggingface_hub<1.0" torchaudio
pip install git+https://github.com/hkchengrex/MMAudio.git
```
