Scripts to experiment with three open-weights text-to-audio models:
* [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0)
* [MMAudio](https://github.com/hkchengrex/MMAudio)
* [AudioLDM2](https://github.com/haoheliu/AudioLDM2)

Each model needs to be installed and run a bit differently, and in separate
Python venvs due to conflicting dependencies. See
[`linux_install.md`](linux_install.md) or
[`macos_install.md`](macos_install.md). Running on MacOS in particular needs a
few patches and workarounds.

Once installed, there are three scripts for each model:
* `*_single.py`: generates N samples for a single prompt (both the prompt and N
  can be set inside the script)
* `*_esc50.py`: Re-runs the ESC-50 experiments from our paper, i.e. 100 samples
  each for the prompt "Sound of [label]" for each label in ESC-50. Note:
  generates 5000 audio files and will take a while!
* `*_param_count.py`: Counts the number of parameters in each model. Added
  because I got frustrated trying to figure out how big the models were from
  their papers (they are all around 1-2b parameters, as it turns out).

Our initial paper:

* Jonathan Morse, Azadeh Naderi, Swen Gaudl, Mark Cartwright, Amy K. Hoover,
  Mark J. Nelson (2025). [Expressive range characterization of open
text-to-audio models](https://doi.org/10.1609/aiide.v21i1.36813). In
*Proceedings of the AAAI Conference on Artificial Intelligence and Interactive
Digital Entertainment*, pp. 91-98.

