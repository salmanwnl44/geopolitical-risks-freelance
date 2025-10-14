# Geopolitical Risks — Freelance

A small demo project that generates synthetic geopolitical-event data, trains a regression model to predict business revenue loss, and provides a simple inference script.

Quick links
- [Both.py](Both.py) — combined training + example usage.
- [data_generater.py](data_generater.py) — synthetic dataset generation (see [`data_generater.num_records`](data_generater.py)).
- [scripts/Train.py](scripts/Train.py) — training script that saves [`scripts.Train.model`](scripts/Train.py) and [`scripts.Train.label_encoders`](scripts/Train.py).
- [scripts/predict.py](scripts/predict.py) — example inference using [`scripts.predict.encode_value`](scripts/predict.py).
- [requirements.txt](requirements.txt) — project dependencies.

Repository overview
- data_generater.py
  - Generates a CSV dataset named `geopolitical_risk_data.csv` with `num_records` synthetic events.
  - Columns include: Event_ID, Date, Country, Event_Type, Risk_Category, Severity_Score, Sentiment_Score, Revenue_Loss%, Supply_Delay_Days, Market_Impact%, Source. See [data_generater.py](data_generater.py).
- scripts/Train.py
  - Loads `geopolitical_risk_data.csv`, encodes categorical columns (stores encoders in `label_encoders`), trains a `RandomForestRegressor`, evaluates and saves the trained [`scripts.Train.model`](scripts/Train.py) and [`scripts.Train.label_encoders`](scripts/Train.py). See [scripts/Train.py](scripts/Train.py).
- scripts/predict.py
  - Loads saved `model.pkl` and `label_encoders.pkl`, encodes inputs via [`scripts.predict.encode_value`](scripts/predict.py), and prints a predicted revenue loss. See [scripts/predict.py](scripts/predict.py).
- Both.py
  - A one-file example that mirrors training and an example prediction using [`Both.model`](Both.py) and encoders created inline. See [Both.py](Both.py).

Setup
1. Create and activate a Python environment (recommended):
```sh
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

Typical workflow
1. Generate data:
```sh
python data_generater.py
```
This writes `geopolitical_risk_data.csv` with [`data_generater.num_records`](data_generater.py) records.

2. Train the model:
```sh
python scripts/Train.py
```
Outputs:
- `model.pkl` (the trained [`scripts.Train.model`](scripts/Train.py))
- `label_encoders.pkl` (the [`scripts.Train.label_encoders`](scripts/Train.py))

3. Run a prediction:
```sh
python scripts/predict.py
```
This uses [`scripts.predict.encode_value`](scripts/predict.py) to prepare inputs and the saved model to predict revenue loss.

Evaluation
The training scripts print common regression metrics: R², MSE and RMSE. RMSE is computed as:

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}
$$

Notes
- Categorical encoders are persisted so inference uses the same mappings as training (see [`scripts.Train.label_encoders`](scripts/Train.py) and [`scripts.predict.encode_value`](scripts/predict.py)).
- Use [Both.py](Both.py) for a compact example combining encoding, training and a sample prediction.
- This workspace is intended as a minimal demo; customize features, model hyperparameters, and evaluation protocols as needed.


