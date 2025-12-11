Scripts to experiment with three open-weights text-to-audio models:
* [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0)
* [MMAudio](https://github.com/hkchengrex/MMAudio)
* [AudioLDM2](https://github.com/haoheliu/AudioLDM2)

Each model needs to be installed and run a bit differently, alas:
* Stable Audio Open is hosted on Huggingface and is run via the `diffusers`
  library. It requires accepting the EULA on Huggingface and putting an HF
authentication token in the environment variable `HF_TOKEN`.
* MMAudio is run by cloning the GitHub repo and installing it as a local
  package in Python
* AudioLDM2 is run by cloning the GitHub repo and using its command-line
  interface

Each model should also use a separate Python venv due to conflicting
dependencies. MacOS needs some additional workarounds, documented in
[`macos_install.md`](macos_install.md).

Once installed, there are three scripts for each model:
* `*_single.py`: generates N samples for a single prompt (both the prompt and N
  can be set inside the script)
* `*_esc50.py`: Re-runs the ESC-50 experiments from our paper, i.e. 100 samples
  each for the prompt "Sound of [label]" for each label in ESC-50. Note:
  generates 5000 audio files and will take a while!
* `*_param_count.py`: Counts the number of parameters in each model. Added
  because I got frustrated trying to figure out how big the models were from
  their papers (they are all about 1b parameters, as it turns out).

Our initial paper:

* Jonathan Morse, Azadeh Naderi, Swen Gaudl, Mark Cartwright, Amy K. Hoover,
  Mark J. Nelson (2025). [Expressive range characterization of open
text-to-audio models](https://doi.org/10.1609/aiide.v21i1.36813). In
*Proceedings of the AAAI Conference on Artificial Intelligence and Interactive
Digital Entertainment*, pp. 91-98.

