# Fast Reconstruction of Exact Maxwell Dynamics from Sparse Data

## Overview of files

### Directories

- `archived/` - Some functions previously experimented with, as well as a function to turn some Maple outputs into PyTorch code (`maple_to_torch_transpile.py`). Additionally, `wave_equation.py` is an older setup working with the 3D wave equation rather than Maxwell's equations.
- `experiments/` - A log of experiments. Not all have been committed due to GitHub's limits.
- `figs/` - Timing plots as in the paper.
- `gifs/` - GIFs of true vs. predicted results reported in the paper. 4 ground-truth solutions x 5 random seeds.
- `neurips_functions/` - the 4 ground-truth solutions used in the paper.

- `notebooks/` - data analyses/most tables reported in the paper.
  - `data_analysis_cpu.ipynb` - used for Table 4. Shell: `run_cpu.sh`
  - `data_analysis.ipynb` - used for Table 1. Shell: `run_experiments.sh`
  - `fem_data_analysis.ipynb` - FEMs, Table 1.
  - `gp_data_analysis.ipynb` - GPs, Table 1.
  - `n_points.ipynb` - used for Table 3. Shell: `run_sensitivity.sh`.
  - `new_sensitivity.ipynb` - more expansive data-capacity trade-off analysis. Not reported in the paper.
  - `noise.ipynb` - experiment prompted by review process (addition of Gaussian noise to training targets for robustness analysis). Shell: `run_noise.sh`
  - `pinn_data_analysis.ipynb` - PINNs, Table 1.
  - `timing_plot_no_clipping.ipynb` - version without cropping out extremely high error values.
  - `timing_plot.ipynb` - used for Figures 2 and 3.
  - `width.ipynb` - used for Table 3. Shell: `run_sensitivity.sh`

- `shells/` - shell scripts used to conduct experiments. May or may not currently reflect the exact setups in the paper.
  - `run_cpu_one_more.sh` - CPU runs w/ BC on solution 2 (the most challenging one).
  - `run_cpu.sh` - Table 4.
  - `run_experiments.sh` - Table 1.
  - `run_noise.sh` - reviewer-requested experiment with noise.
  - `run_sensitivity.sh` - various additional analyses (e.g. Table 3).
  - `run_viz.sh` - used to produce `gifs/`.

- `tools/` - Python scripts for searching through experimental results.
  - `analyze_run.py` - prints a brief analysis of one or more training runs in the `experiments/` directory. Example: `python analyze_run.py -s 1568 --soln 3` (shows runs of solution 3 starting from experiment 1568).
  - `compare_runs.py` - compares two experimental runs in the `experiments/` directory in terms of data and hyperparameters. Example: `python compare_runs.py -s 1568 -e 1569` (compares these two runs).
  - `search_runs.py` - implements a MongoDB like matching scheme on `hparams.json` files to search for specific setups by filtering.

### Main project Python scripts

- `data_sampler.py` - implements on-the-fly dataset generation for IC and BC setups, with an option to uniformly sample from a domain or to build a grid over that domain.
- `function.py` - should contain a function `u(t: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tensor` where `t`, `x`, `y`, `z` are each of shape `[1, n]`. The result are the E and B fields evaluated at the input spacetime coordinates, of shape `[6, n]` (Ex, Ey, Ez, Bx, By, Bz).
- `maxwell_equation.py` - the architecture of FLASH-MAX, fully specified in `torch`. Can run the `main()` method to test functionality.
- `symlog.py` - alternative loss function utilities. Not used in the FLASH-MAX paper.
- `train.py` - script to train a FLASH-MAX model. Can learn a custom function defined in `function.py`, or one of the 4 predefined solutions with `--soln=x` for `x` in `{1, 2, 3, 4}`. Has many other command-line arguments to set IC, BC, training and validation domains, data quantity, and various training hyperparameters.
- `utils.py` - `PCNN` superclass for extending the project further and `get_device`.
- `visualize.py` - produces quiver plots gifs, weight heatmaps, and other visualizations depending on use case. Without an argument, visualizes most recent experiment. Can also be used like: `py visualize.py -e 1568` to visualize experiment 1568.

### Other

- `.gitignore` - as per usual.
- `README.md` - this file.
- `requirements.txt` - Python environment for FLASH-MAX. Works with `conda`.
