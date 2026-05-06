# Shallow neural networks for Maxwell's equations

## Overview of files

### Python scripts

- `analyze_run.py` - prints a brief analysis of one or more training runs in the `experiments/` directory. Example: `python analyze_run.py -s 1568 --soln 3` (shows runs of solution 3 starting from experiment 1568).

- `compare_runs.py` - compares two experimental runs in the `experiments/` directory in terms of data and hyperparameters. Example: `python compare_runs.py -s 1568 -e 1569` (compares these two runs).

- `data_sampler.py` - implements on-the-fly dataset generation for IC and BC setups, with an option to uniformly sample from a domain or to build a grid over that domain.

- `function.py` - should contain a function `u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor` where `t`, `x`, `y`, `z` are each of shape `[1, n]`. The result are the E and B fields evaluated at the input spacetime coordinates, of shape `[6, n]` (Ex, Ey, Ez, Bx, By, Bz).

- `maple_to_torch_transpile.py` - expects a Maple output list of functions, and will attempty to reimplement those functions in `torch`.

- `maxwell_equation.py` - the architecture of FLASH-MAX, fully specified in `torch`. Can run the `main()` method to test functionality.

- `move_gifs.py` - move some representative gifs into a dedicated directory `flash_max_gifs/` for ease of access.

- `symlog.py` - alternative loss function utilities. Not used in the FLASH-MAX paper.

- `train.py` - script to train a FLASH-MAX model. Can learn a custom function defined in `function.py`, or one of the 4 predefined solutions with `--soln=x` for `x` in `{1, 2, 3, 4}`. Has many other command-line arguments to set IC, BC, training and validation domains, data quantity, and various training hyperparameters.

- `utils.py` - miscellaneous.

- `visualize.py` - produces quiver plots gifs, weight heatmaps, and other visualizations depending on use case. Without an argument, visualizes most recent experiment. Can also be used like: `py visualize.py -e 1568` to visualize experiment 1568.

- `wave_equation.py` - older setup working with the 3D wave equation rather than Maxwell's equations.

### IPython notebooks and associated shell scripts that generated the data

- `data_analysis_cpu.ipynb` - used for Table 4. Shell: `run_cpu.sh`

- `data_analysis.ipynb` - used for Table 1. Shell: `run_experiments.sh`

- `n_points.ipynb` - used for Table 3. Shell: `run_n_points.sh`

- `timing_plot.ipynb` - used for Figures 2 and 3.

- `width.ipynb` - used for Table 3. Shell: `run_sensitivity.sh`

### Other

- `README.md` - this file.

- `requirements.txt` - Python environment for FLASH-MAX. Works with `conda`.

- `run_quick_and_analyze.sh` - quick tests for hyperparameter tuning.

- `neurips_functions/` - contains the four ground truth solutions used in the FLASH-MAX paper.

- Other directories are largely just folders of scripts containing possible functions.
