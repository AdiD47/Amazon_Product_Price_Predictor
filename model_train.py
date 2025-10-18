import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from typing import Tuple, Dict
from uitil import calculate_regression_metrics


class XGBoostPricePredictor:
    """
    XGBoost-based price prediction model with cross-validation.
    Applies log-transform to target before training
    and inverse transform (exp) on predictions.
    """

    def __init__(self, params: Dict = None, n_folds: int = 5):
        self.n_folds = n_folds
        self.models = []
        self.oof_predictions = None
        self.feature_importance = None

        self.params = params or {
            "objective": "reg:squarederror",
            "learning_rate": 0.01,
            "max_depth": 8,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,  # L1
            "reg_lambda": 0.8,  # L2
            "n_estimators": 3000,
            "tree_method": "hist",
            "early_stopping_rounds": 200
        }

    def train_with_cv(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Train model using K-Fold cross-validation with log-transformed target.
        """
        # ✅ Apply log transformation
        y_log = np.log1p(y)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        self.oof_predictions = np.zeros(len(X))
        fold_metrics = []

        print(f"\nTraining XGBoost with {self.n_folds}-Fold Cross-Validation...")
        print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            print(f"\n{'='*50}")
            print(f"Fold {fold}/{self.n_folds}")
            print(f"{'='*50}")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y_log[train_idx], y_log[val_idx]

            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.params.get('n_estimators', 1000),
                evals=[(dval, 'eval')],
                early_stopping_rounds=50,
                verbose_eval=False
            )

            val_predictions_log = model.predict(dval, iteration_range=(0, model.best_iteration))
            val_predictions = np.expm1(val_predictions_log)  # ✅ Inverse transform
            val_predictions = np.maximum(val_predictions, 0.01)

            self.oof_predictions[val_idx] = val_predictions

            metrics = calculate_regression_metrics(y[val_idx], val_predictions)
            fold_metrics.append(metrics)

            print(f"Fold {fold} SMAPE: {metrics['smape']:.4f} | "
                  f"MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | "
                  f"R²: {metrics['r2']:.4f}")
            print(f"Best iteration: {model.best_iteration}")

            self.models.append(model)

        overall_metrics = calculate_regression_metrics(y, self.oof_predictions)

        print(f"\n{'='*50}")
        print(f"Cross-Validation Results:")
        print(f"{'='*50}")

        for metric_name in fold_metrics[0].keys():
            mean_metric = np.mean([m[metric_name] for m in fold_metrics])
            std_metric = np.std([m[metric_name] for m in fold_metrics])
            print(f"Mean {metric_name.upper()}: {mean_metric:.4f} (±{std_metric:.4f})")

        print(f"\nOverall OOF Metrics:")
        for name, value in overall_metrics.items():
            print(f"  - {name.upper()}: {value:.4f}")

        print(f"{'='*50}\n")

        self._calculate_feature_importance(X)

        return overall_metrics, self.oof_predictions

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ensemble of trained models.
        Applies inverse log transform (exp).
        """
        if not self.models:
            raise ValueError("Model not trained. Call train_with_cv first.")

        predictions_log = np.zeros(len(X))

        for model in self.models:
            dtest = xgb.DMatrix(X)
            predictions_log += model.predict(dtest, iteration_range=(0, model.best_iteration))

        predictions_log /= len(self.models)

        # ✅ Inverse log transform
        predictions = np.expm1(predictions_log)
        predictions = np.maximum(predictions, 0.01)

        return predictions

    def _calculate_feature_importance(self, X: pd.DataFrame):
        """
        Placeholder for feature importance.
        """
        pass

    def get_top_features(self, top_n: int = 20) -> pd.DataFrame:
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated. Train model first.")
        return self.feature_importance.head(top_n)


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: Dict = None,
    n_folds: int = 5
) -> XGBoostPricePredictor:
    """
    Train XGBoost model with optimized hyperparameters.
    """
    model = XGBoostPricePredictor(params=params, n_folds=n_folds)
    model.train_with_cv(X_train, y_train)
    return model
