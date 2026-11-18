# Mango Remaining Shelf-Life Prediction — MLOps Application

A runnable academic project for predicting remaining marketable shelf life (days) of mango shipments arriving at a packing facility near Nuzvid, Andhra Pradesh.

## Features
- 600 synthetic shipment records: 300 baseline Version 1 and 300 delayed-transport/cooling Version 2.
- Reusable feature files with `heat_exposure_index` and `handling_damage_index`.
- Random Forest and Gradient Boosting regression comparison (MAE, RMSE, R²).
- MLflow experiment tracking and model registry when MLflow is installed.
- FastAPI backend plus a browser-based prediction form.
- Pydantic input validation and GitHub Actions CI tests.

## Run on Windows (VS Code terminal)
Use Python 3.11 or 3.12.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.generate_data
python -m src.features
python -m src.train
python -m pytest -q
uvicorn src.api:app --reload
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

Open `http://127.0.0.1:8000` for the web application and `http://127.0.0.1:8000/docs` for Swagger API docs. Alternatively, double-click `setup_and_run.bat` to create the environment, install packages, generate data, train, test and launch the application.

## MLflow dashboard
In another terminal:
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open `http://127.0.0.1:5000`. Training logs both models' parameters/metrics/artifacts and registers the selected model as `MangoShelfLifeRegressor`. MLflow is optional for local model training, but install the listed requirements to enable tracking/registry.

## API example
`POST http://127.0.0.1:8000/predict`
```json
{
  "harvest_maturity_index": 0.72,
  "pulp_temperature_c": 22.5,
  "brix": 14.2,
  "bruising_percentage": 3.5,
  "transport_duration_hours": 12,
  "cooling_delay_hours": 2,
  "packaging_type": "ventilated_crate"
}
```
The response contains `estimated_remaining_shelf_life_days` and the selected model name.

## Dataset and engineered features
- `data/mango_shipments_v1.csv`: normal baseline transport/cooling profile.
- `data/mango_shipments_v2.csv`: increased transport and cooling delays.
- `data/features/`: reusable processed feature CSVs.
- `heat_exposure_index = max(pulp_temperature_c - 12, 0) * transport_duration_hours / 24`
- `handling_damage_index = bruising_percentage * 0.08 + cooling_delay_hours * 0.55`

## Important limitation
The data and target labels are synthetic, produced from an illustrative relationship plus random variation. This project demonstrates the MLOps lifecycle and is not validated for real export decisions. Collect real shipment, cultivar, maturity, cold-chain and marketability inspection data and validate the model before operational use.

## Verified local demonstration results

On the generated 600-row synthetic dataset and a fixed 80/20 split, the included baseline run selected Random Forest (MAE 1.538 days, RMSE 1.934 days, R² 0.824). Gradient Boosting scored MAE 1.540 days, RMSE 1.973 days, R² 0.817. Results can vary if code, package versions, random seed or data generation changes.
