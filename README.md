# stable-baselines3-extensions
Stable-Baselines3 Extensions contains custom extensions for Stable-Baselines3 that are used by our research group.

The main differences compared to the latest release of Stable-Baselines3 are:
- EvalCallback:
    - if an evaluation file already exists under the specified log_path, the data will be loaded
      (makes it possible to continue training)
- Adds a linear learning rate schedule. (based on rl-baselines3-zoo implementation) 
- Prints a more detailed system info
- Hindsight Experience Replay:
    - the implementation contained in this package can also relabel desired goals in the info dictionary

## Installation
This package can be installed using:
```
cd PATH_TO_THIS_PACKAGE
pip install -e .
```
or
```
cd PATH_TO_THIS_PACKAGE
pip install -e .[tests]
```
**Note:** Depending on your shell (e.g. when using Zsh), you may need to use additional quotation marks: 
```
cd PATH_TO_THIS_PACKAGE
pip install -e ."[tests]"
```

## Maintainer
Stable-Baselines3-Extensions is currently maintained by Lara Bergmann (@lbergmann1).