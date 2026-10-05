# DFT validation

`scripts/dft_stability.py` checks MLFF stability numbers with DFT. It uses pymatgen and custodian (no workflow
database) and Materials-Project-compatible settings: `MPRelaxSet` relax, `MPRelaxSet` relax again, `MPStaticSet`
energy, then `MaterialsProject2020Compatibility` corrections and the energy above the **full** Materials Project
GGA/GGA+U hull from the API.

## One-time setup

```bash
pip install -e ".[dft]"
pmg config --add PMG_VASP_PSP_DIR /path/to/pymatgen_potcars     # POTCARs, licensed VASP required
export MP_API_KEY=...                                           # free key, never commit it
export VASP_CMD="srun -n 32 vasp_std"
```

## Workflow

```bash
python scripts/dft_stability.py select                # choose structures from the evaluation JSON
python scripts/dft_stability.py run --dry-run --pos 0 # check the VASP inputs for one structure
python scripts/dft_stability.py slurm                 # writes submit_dft.sbatch (or: run --all --workers N)
python scripts/dft_stability.py status
python scripts/dft_stability.py analyze --novelty-ref mp20_train_structures.json
```

`select` needs the relaxed structures inside the evaluation JSON: have the evaluation script store
`out["structures"] = [dict(idx=..., structure=<Structure.as_dict()>), ...]` before it discards them.

By default `select` takes every structure whose MLFF energy above the hull is at most 0.2 eV/atom plus a random
control sample from above that cut, so the screen itself can be tested. `analyze` reports rates over all sampled
structures and, separately, an extrapolation that uses the control sample.
