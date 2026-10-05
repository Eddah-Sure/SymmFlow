# Data preparation

**TODO before publishing:** add the scripts that turn the raw MP-20 CSV files into the `train/val/test`
folders described in `../data/README.md` (the ones that mine the asymmetric unit, Wyckoff data and graphs and
write `<split>.npz` / `<split>_config.json`).

Checklist when you add them:

1. Take input and output locations from command-line arguments or from `MP20_ROOT`; do not hard-code them.
2. Run `python tools/check_private_paths.py` (it flags machine-specific paths, e-mail addresses and keys).
3. Document the exact command that regenerates the dataset, and the versions of `pymatgen` and `spglib` used.
