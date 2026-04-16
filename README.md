# LWCUP

Lightweight factor pipeline + LightGBM baseline for LWCUP style tick data.

## Overview
- Factor calculation: `factor_pool/` (tick and daily factor builders).
- Training: LightGBM baseline in `model/`.
- Data prep + entry points: `proc_data.py`, `main.py`, `model/train.py`.

## Project Layout
- `factor_pool/`
	- `tick_factor_pool.py`: tick-level factor registry and pool builder.
	- `daily_factor_pool.py`: daily factor registry and pool builder.
	- `pipeline.py`: orchestrates factor calculation across dates.
	- `FactorEval.py`: factor evaluation (IC/IR, correlation).
- `model/`
	- `lgbm.py`: LightGBM training, evaluation, and model export.
	- `train.py`: example training entry point.
- `data/`: raw parquet files (tick data).
- `results/`: merged data, logs, models, and test data outputs.

## Environment
Install Python dependencies as needed for pandas, numpy, lightgbm, optuna, sklearn, tqdm.

## Data Preparation
Generate merged parquet and test data:

```bash
python proc_data.py
```

Outputs:
- `results/merge_data/merge_data.parquet`
- `results/test_data/test_data.pkl`

## Factor Registry
Factor definitions live in `config.py` under `FACTORS`.
Each factor entry has:

```python
{
		"factor_func": <callable>,
		"need_cols": ["col1", "col2", ...]
}
```

Example (from `config.py`):

```python
FACTORS = {
		"tick_OBI": {
				"factor_func": tick_Orderbook_Imbalance_single_day,
				"need_cols": ["bid1", "ask1", "bsize1", "asize1"],
		},
}
```

## Train LightGBM
Two entry points are provided.

Option A: training script in `model/train.py`:

```bash
python model/train.py
```

Option B: CLI wrapper in `main.py`:

```bash
python main.py --task label_20 --factors tick_OBI
```

Artifacts go to:
- `results/lgbm_logs/`
- `results/lgbm_models/`

## Factor Evaluation
Use `factor_pool/FactorEval.py` to compute IC/IR and factor correlation on merged data.

```bash
python factor_pool/FactorEval.py
```

## Notes
- `config.py` contains data paths, date ranges, and factor registry.
- `results/merge_data/null_sym_date_cols.json` can be used to drop null dates during training.
