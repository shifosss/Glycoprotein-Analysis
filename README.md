# Glycoprotein-Analysis

Machine-learning code for predicting protein–glycan binding strength: it compares glycan and protein encoders and the regression models built on them. It was written during a research internship at Shenzhen Bay Laboratory, June to August 2025.

## What it contains

- **Data**: `data/v12_glycan_binding.csv` has 2,122 rows, one per protein sequence entry (2,097 distinct `target` sequences; 1,386 distinct `protein` names, with 361 rows unnamed), and 2,880 columns: 2,878 glycan columns holding binding values (799,408 filled cells), plus `target` (the protein sequence) and `protein` (the protein name).
- **Exploration**: `Data Preprocessing and Analysis.ipynb` loads the interaction table from `glycowork` and profiles it. One cell reads `data/glycan_interaction.csv`, which is not in the repo; supply that file to rerun it.
- **Embedders** (`embedder/`): ESM-2 protein embeddings (650M or 3B weights, via `fair-esm`) and glycan embeddings (`lstm` by default; `gcn`, `bert`, `cnn`, `rgcn`, `gat` and `gin` are listed in `Customization_Guide`), combined by concatenation or attention fusion. `embedder/GlycanEmbedder_Package/` holds the glycan encoders with `lstm_embedder.pth`, `gcn_embedder.pth` and the vocabulary.
- **Splits** (`preprocessing/`): `clustering_splitter.py` clusters proteins with K-means on their embeddings (10 clusters by default) and assigns whole clusters to train, validation and test, so held-out proteins are not near-copies of training ones. `embedding_preprocessor.py` precomputes and caches the embeddings.
- **Networks** (`network/`): `mlp`, `residual_mlp`, `attention` and `ensemble` regressors.
- **Training scripts**: `training_example_cpu_v2.py` (precomputed embeddings, clustered splits; the default run is an `mlp` with hidden sizes 1024, 512, 256, 128), `gpu_optim_training_script.py` (multi-worker loading for a GPU) and older variants in `training_example/` and `deprecated/`. Training reports validation and test R².
- `Customization_Guide` (in Chinese) lists the settings you can change in the training script.

## Running it

Prerequisites: Python 3.10, a CUDA GPU for the 650M ESM-2 model (the training scripts use CUDA when available and run on CPU otherwise), and the packages in `embedder/GlycanEmbedder_Package/requirements.txt` plus `fair-esm`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn` and `tqdm`. The full pinned environments are `env_setting/environment.yml` and `environment_autodl.yml` (UTF-16 conda exports from Windows).

```bash
python training_example_cpu_v2.py              # embed, split by cluster, train, evaluate
python training_example_cpu_v2.py precompute   # only build the embedding cache
```

Not verified: these commands have not been run from a clean clone. The scripts use relative paths that depend on where they are launched (`resources/esm-model-weights` for the ESM-2 weights, and `GlycanEmbedder_Package/glycoword_vocab.pkl` for the vocabulary, which sits under `embedder/` in this repo). `dataloader/glycan_dataloader_cpu_v2.py` also imports `embedding_preprocessor` and `clustering_splitter` as top-level modules, so `preprocessing/` must be on `PYTHONPATH`. Expect to adjust these.

## Status

The repo holds code and data only. It contains no saved results or trained predictor, so it shows no benchmark numbers.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
