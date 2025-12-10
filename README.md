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
dependencies.

Our initial paper:

* Jonathan Morse, Azadeh Naderi, Swen Gaudl, Mark Cartwright, Amy K. Hoover,
  Mark J. Nelson (2025). [Expressive range characterization of open
text-to-audio models](https://doi.org/10.1609/aiide.v21i1.36813). In
*Proceedings of the AAAI Conference on Artificial Intelligence and Interactive
Digital Entertainment*, pp. 91-98.

