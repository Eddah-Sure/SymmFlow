# Scripts

| Script | Purpose | Needs |
|---|---|---|
| `visualize.py` | Draw generation trajectories, or (`--panel`) a figure of crystals generated under requested space groups with the asymmetric unit and Wyckoff labels | trained checkpoint, `symfix` |
| `make_figures.py` | Draw the results figures from an evaluation JSON | evaluation JSON |
| `dft_stability.py` | Select structures, run VASP (relax, relax, static), compute energy above the full Materials Project hull | VASP, POTCARs, MP API key |
| `evaluate.py` | **To be added:** the evaluation script that writes the evaluation JSON | trained checkpoint, dataset |

`evaluate.py` was not part of the initial import. Add it here, take its dataset and checkpoint locations from
`MP20_ROOT` / `EMF_CHECKPOINT`, and run `python tools/check_private_paths.py` before committing.

All scripts read their locations from environment variables (see `../.env.example`) or command-line flags.
