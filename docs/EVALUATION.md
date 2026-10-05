# Evaluation

The evaluation script (to be added as `scripts/evaluate.py`) loads a checkpoint, **installs the symmetry
correction (`symfix`)**, generates 10,000 structures and writes one JSON file. `scripts/make_figures.py` reads only
that JSON, so every figure and table value traces back to a single file.

## Quantities in the JSON that the figure script uses

| Key | Meaning |
|---|---|
| `n_generated`, `n_struct_valid`, `n_comp_valid` | sample count, 0.5 A distance test, SMACT composition test |
| `model_gate_validity` | stricter training-time criterion (radius exclusion, VPA, connectivity) |
| `coverage.ds0.2/0.4/0.6` | COV-R and COV-P at three threshold pairs |
| `symmetry`, `site_occupancy`, `property_distances` | space-group, Wyckoff and density statistics |
| `conditional_fidelity.by_guidance` | per-target surrogate and independent-oracle errors, per guidance scale |
| `sg_controllability.rows` | requested-group exact match and the shuffled-conditioning control |
| `stability` | MatterSim relaxation records and summary |

## Things to state whenever numbers are reported

* **Coverage protocol.** Published baselines compute coverage over *valid* generated structures only. Record
  whether your run did the same (`valid_only`), and do not compare coverage across the two.
* **Hull.** A hull built from MP-20 compounds only is a *subset* of the Materials Project hull, so energies above it
  are lower bounds and stable fractions are upper bounds. Confirm with DFT (`docs/DFT.md`).
* **Energy oracle.** State its hold-out error; differences below that error are not resolvable.
* **Shuffled-conditioning control.** The exact-match rate for a requested space group is guaranteed by orbit
  expansion; only the control separates this from learned conditioning.
* **One sampling seed.** Report intervals (e.g. Wilson) and the number of samples behind each rate.
