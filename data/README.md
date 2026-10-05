# Data

Nothing in this folder is tracked by git (see `.gitignore`). Put the mined MP-20 dataset here, or keep it
anywhere you like and point the code at it with an environment variable:

```bash
export MP20_ROOT=/path/to/MP20        # the folder that contains train/ val/ test/
```

No machine-specific path is stored in the repository; `MP20_ROOT` (default: `data/MP20`) is the only way the
code locates the data.

## Expected layout

```
$MP20_ROOT/
├── train/
│   ├── train.npz            # key "graph_dict": {material_id: {nodes, frac_coords, cart_coords,
│   │                        #   lattice_matrix, space_group, Wyckoff/orbit info, ...}}
│   └── train_config.json    # "atomic_numbers", optional "element_properties", "node_vectors"
├── val/   (same files, val.npz / val_config.json)
└── test/  (same files, test.npz / test_config.json)
```

`sitetokens.MultiFileGraphDataset` reads exactly these files (`<split>/<split>.npz` and
`<split>/<split>_config.json`). The `.npz` files are produced from the raw MP-20 CSVs by the data-preparation
scripts (see `../data_prep/README.md`).

## Source of MP-20

MP-20 is the 45,231-structure benchmark introduced with CDVAE (Xie et al., 2022): crystals from the Materials
Project with at most 20 atoms in the cell. The raw CSV splits are in the CDVAE repository (`data/mp_20`).
Respect the licences of the Materials Project data and of the CDVAE repository when redistributing anything
derived from them; this repository deliberately does **not** ship any data.
