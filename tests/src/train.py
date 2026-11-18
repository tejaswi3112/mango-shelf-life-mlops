import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import FEATURES, TARGET, DATA_DIR, ARTIFACT_DIR
from src.features import engineer_features


def train():
    raw = pd.concat([
        pd.read_csv(DATA_DIR / "mango_shipments_v1.csv"),
        pd.read_csv(DATA_DIR / "mango_shipments_v2.csv")
    ], ignore_index=True)
    data = engineer_features(raw)
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    estimators = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.04, max_depth=2,
            loss="huber", random_state=42
        )
    }
    results, best = [], None
    mlflow_available = False
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("mango-shelf-life")
        mlflow_available = True
    except ImportError:
        print("MLflow is not installed; training locally without experiment tracking.")

    for name, estimator in estimators.items():
        model = Pipeline([("scaler", StandardScaler()), ("regressor", estimator)])
        run_context = mlflow.start_run(run_name=name) if mlflow_available else None
        if run_context:
            run_context.__enter__()
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            metrics = {
                "model": name,
                "MAE": float(mean_absolute_error(y_test, pred)),
                "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
                "R2": float(r2_score(y_test, pred))
            }
            results.append(metrics)
            if mlflow_available:
                mlflow.log_param("model", name)
                mlflow.log_param("dataset", "V1+V2")
                mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
                mlflow.sklearn.log_model(model, artifact_path="model")
                run_id = mlflow.active_run().info.run_id
            else:
                run_id = None
            if best is None or metrics["MAE"] < best["metrics"]["MAE"]:
                best = {"name": name, "model": model, "metrics": metrics, "run_id": run_id}
        finally:
            if run_context:
                run_context.__exit__(None, None, None)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(best["model"], ARTIFACT_DIR / "best_model.joblib")
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps({
        "selected_model": best["name"], "features": FEATURES,
        "run_id": best["run_id"]
    }, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)

    if mlflow_available and best["run_id"]:
        try:
            mlflow.register_model(f"runs:/{best['run_id']}/model", "MangoShelfLifeRegressor")
        except Exception as exc:
            print(f"MLflow registry registration warning: {exc}")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"Selected model: {best['name']}")
    print(f"Saved model: {ARTIFACT_DIR / 'best_model.joblib'}")
    return results


if __name__ == "__main__":
    train()
