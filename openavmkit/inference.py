from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import warnings
import geopandas as gpd
from openavmkit.filters import select_filter
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split, KFold
import itertools
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize

class InferenceModel(ABC):
    """Base class for inference models"""
    
    @abstractmethod
    def fit(self, df: pd.DataFrame, target: str, settings: Dict[str, Any]) -> None:
        """Fit the model using training data"""
        pass
        
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Make predictions on new data"""
        pass
        
    @abstractmethod
    def evaluate(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Evaluate model performance on training data"""
        pass

class RatioProxyModel(InferenceModel):
    """Ratio-based proxy model with proper validation handling"""
    
    def __init__(self):
        self.proxy_ratios = {}
        self.proxy_stats = {}
        
    def fit(self, df: pd.DataFrame, target: str, settings: Dict[str, Any]) -> None:
        """Fit ratio model handling missing values"""
        proxies = settings.get("proxies", [])
        locations = settings.get("locations", [])
        group_by = settings.get("group_by", [])
        
        # Add global grouping
        locations.append("___everything___") 
        df["___everything___"] = "1"
        
        self.proxy_ratios = {}
        self.proxy_stats = {}
        
        # Calculate ratios for each proxy
        for proxy in proxies:
            # Handle missing values
            valid_mask = (
                df[target].notna() & 
                df[proxy].notna() & 
                df[proxy].gt(0) &
                df[target].gt(0)
            )
                        
            if valid_mask.sum() == 0:
                warnings.warn(f"No valid data for proxy {proxy}")
                continue
                
            # Calculate ratios
            df_valid = df[valid_mask].copy()
            df_valid[f"ratio_{proxy}"] = df_valid[target] / df_valid[proxy]
            
            # Remove outliers
            q1, q99 = df_valid[f"ratio_{proxy}"].quantile([0.01, 0.99])
            valid_range = (df_valid[f"ratio_{proxy}"] >= q1) & (df_valid[f"ratio_{proxy}"] <= q99)
            df_valid = df_valid[valid_range]
            
            # Store global ratio
            global_ratio = df_valid[f"ratio_{proxy}"].median()
            self.proxy_ratios[(proxy, ())] = global_ratio
            
            # Calculate ratios for each location/group combination
            for location in locations:
                if location == "___everything___":
                    continue
                    
                group_list = group_by.copy() if group_by else []
                group_list.append(location)
                
                try:
                    grouped = df_valid.groupby(group_list)
                    median_ratios = grouped[f"ratio_{proxy}"].median()
                    if not median_ratios.empty:
                        self.proxy_ratios[(proxy, tuple(group_list))] = median_ratios
                except Exception as e:
                    warnings.warn(f"Failed to calculate grouped ratios for {proxy} with groups {group_list}: {str(e)}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Make predictions using fitted ratios
        """
        predictions = pd.Series(index=df.index, dtype='float64')
        
        for (proxy, group_list) in sorted(self.proxy_ratios.keys(), key=lambda x: len(x[1]), reverse=True):
            if len(group_list) > 0:
                try:
                    # Get group-specific ratios
                    group_key = df[list(group_list)].astype(str).agg('_'.join, axis=1)
                    ratios = self.proxy_ratios[(proxy, group_list)]
                    
                    # Only apply ratios for existing group combinations
                    common_keys = group_key[group_key.isin(ratios.index)]
                    if not common_keys.empty:
                        # Create initial mask
                        mask = (predictions.isna() & 
                               df[proxy].notna() & 
                               df[proxy].gt(0) & 
                               group_key.isin(ratios.index))
                               
                        # Additional validation
                        proxy_values = df.loc[mask, proxy]
                        ratio_values = ratios[group_key[mask]]
                        predicted_values = ratio_values * proxy_values
                        
                        # Create validation mask aligned with original mask
                        valid_predictions = pd.Series(False, index=df.index)
                        valid_predictions.loc[mask] = (predicted_values > 100) & (predicted_values < 100000)
                        
                        # Combine masks
                        final_mask = mask & valid_predictions
                        
                        # Apply predictions
                        predictions.loc[final_mask] = predicted_values[valid_predictions[mask]]
                except Exception as e:
                    warnings.warn(f"Failed to apply grouped ratios for {proxy} with groups {group_list}: {str(e)}")
            else:
                # Apply global ratio to remaining missing values
                ratio = self.proxy_ratios[(proxy, ())]
                mask = predictions.isna() & df[proxy].notna() & df[proxy].gt(0)
                
                # Additional validation for global ratio
                proxy_values = df.loc[mask, proxy]
                predicted_values = ratio * proxy_values
                
                # Create validation mask aligned with original mask
                valid_predictions = pd.Series(False, index=df.index)
                valid_predictions.loc[mask] = (predicted_values > 100) & (predicted_values < 100000)
                
                # Combine masks
                final_mask = mask & valid_predictions
                
                # Apply predictions
                predictions.loc[final_mask] = ratio * df.loc[final_mask, proxy]
                
        return predictions
        
    def evaluate(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """
        Evaluate model performance on training data
        
        Args:
            df: Training DataFrame
            target: Target field name
            
        Returns:
            Dict of performance metrics
        """
        valid_mask = df[target].notna()
        predictions = self.predict(df[valid_mask])
        
        actuals = df.loc[valid_mask, target]
        
        # Calculate metrics
        metrics = {
            'mae': np.abs(predictions - actuals).mean(),
            'mape': np.abs((predictions - actuals) / actuals).mean() * 100,
            'rmse': np.sqrt(((predictions - actuals) ** 2).mean()),
            'r2': 1 - ((predictions - actuals) ** 2).sum() / ((actuals - actuals.mean()) ** 2).sum()
        }
        
        return metrics

class CategoricalEncoder:
    """Universal categorical encoder that handles unseen categories"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.unknown_value = None  # Will be set during fit
        
    def fit(self, series: pd.Series) -> None:
        """Fit encoder adding a special unknown value"""
        # Get unique non-null values
        unique_values = series.dropna().unique()
        
        # Add an extra category for unknown
        self.label_encoder.fit(np.append(unique_values, ['__UNKNOWN__']))
        self.unknown_value = self.label_encoder.transform(['__UNKNOWN__'])[0]
        
    def transform(self, series: pd.Series) -> np.ndarray:
        """Transform values, mapping unseen categories to unknown"""
        # Handle nulls first
        series = series.fillna('__UNKNOWN__')
        
        # Create output array
        result = np.full(len(series), self.unknown_value)
        
        # Get mask of known categories
        known_mask = series.isin(self.label_encoder.classes_)
        
        # Transform known categories
        if known_mask.any():
            result[known_mask] = self.label_encoder.transform(series[known_mask])
            
        return result
        
    def fit_transform(self, series: pd.Series) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(series)
        return self.transform(series)

class SmartOLSModel(InferenceModel):
    """OLS model with universal categorical handling"""
    
    def __init__(self):
        self.best_model = None
        self.best_locations = None
        self.best_score = float('-inf')
        self.encoders = {}
        self.proxy_fields = None
        self.imputer = SimpleImputer(strategy='median')
        
    def _try_location_combination(self, df_train: pd.DataFrame, df_val: pd.DataFrame, 
                                target: str, proxies: List[str], locations: List[str]) -> tuple:
        """Try location combination with proper train/val split"""
        try:
            # Create feature matrices
            X_train = df_train[proxies].copy()
            X_val = df_val[proxies].copy()
            
            # Get target values
            y_train = df_train[target].values
            y_val = df_val[target].values
            
            # Handle missing values in proxy variables
            X_train = pd.DataFrame(self.imputer.fit_transform(X_train), 
                                 columns=X_train.columns, index=X_train.index)
            X_val = pd.DataFrame(self.imputer.transform(X_val), 
                               columns=X_val.columns, index=X_val.index)
            
            # Encode location variables
            for loc in locations:
                if loc not in self.encoders:
                    self.encoders[loc] = CategoricalEncoder()
                    X_train[loc] = self.encoders[loc].fit_transform(df_train[loc])
                else:
                    X_train[loc] = self.encoders[loc].transform(df_train[loc])
                    
                X_val[loc] = self.encoders[loc].transform(df_val[loc])
            
            # Fit model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_score = r2_score(y_val, model.predict(X_val))
            print(f"--> Locations {tuple(locations)}: R² = {val_score:.4f}")
            
            if val_score > self.best_score:
                return model, locations, val_score
                
            return None, None, val_score
            
        except Exception as e:
            print(f"Error trying location combination {locations}: {str(e)}")
            return None, None, float('-inf')

    def fit(self, df: pd.DataFrame, target: str, settings: Dict[str, Any]) -> None:
        """
        Fit model trying different location combinations
        """
        proxies = settings.get("proxies", [])
        locations = settings.get("locations", [])
        
        if not proxies:
            raise ValueError("No proxy fields specified")
            
        self.proxy_fields = proxies
        
        print("\nTrying different location combinations for OLS:")
        
        # Try different combinations of location variables
        for r in range(len(locations) + 1):
            for loc_combo in itertools.combinations(locations, r):
                model, locs, score = self._try_location_combination(
                    df, df, target, proxies, list(loc_combo)
                )
                
                print(f"--> Locations {loc_combo}: R² = {score:.4f}")
                
                if model is not None:
                    self.best_model = model
                    self.best_locations = locs
                    self.best_score = score
        
        print(f"\nBest OLS model:")
        print(f"--> Locations: {self.best_locations}")
        print(f"--> R² Score: {self.best_score:.4f}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Make predictions using best model"""
        if self.best_model is None:
            raise ValueError("Model not fitted")
            
        X = df[self.proxy_fields].copy()
        
        # Handle missing values
        X = pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)
        
        # Add encoded location variables
        if self.best_locations:
            for loc in self.best_locations:
                loc_values = df[loc].fillna('UNKNOWN').astype(str)
                X[loc] = self.encoders[loc].transform(loc_values)
                
        return pd.Series(self.best_model.predict(X), index=df.index)

    def evaluate(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(df)
        actuals = df[target]
        
        metrics = {
            'mae': np.abs(predictions - actuals).mean(),
            'mape': np.abs((predictions - actuals) / actuals).mean() * 100,
            'rmse': np.sqrt(((predictions - actuals) ** 2).mean()),
            'r2': self.best_score
        }
        
        return metrics

class RandomForestModel(InferenceModel):
    """Random Forest with improved validation and parameters"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.encoders = {}
        self.proxy_fields = None
        self.location_fields = None
        self.interaction_fields = None
        self.imputer = SimpleImputer(strategy='median')
        self.feature_order = None  # Store the order of features from training
    
    def _create_feature_matrix(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Create feature matrix with consistent feature order"""
        # Start with proxy fields
        X = df[self.proxy_fields].copy()
        
        # Handle missing values in numeric variables
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            if fit:
                X[numeric_cols] = self.imputer.fit_transform(X[numeric_cols])
            else:
                X[numeric_cols] = self.imputer.transform(X[numeric_cols])
        
        # Add individual location fields
        for loc in self.location_fields:
            if loc in df.columns:
                if fit:
                    self.encoders[loc] = CategoricalEncoder()
                    X[loc] = self.encoders[loc].fit_transform(df[loc])
                else:
                    if loc in self.encoders:
                        X[loc] = self.encoders[loc].transform(df[loc])
        
        # Add interaction features
        if hasattr(self, 'interaction_fields') and self.interaction_fields:
            for interaction in self.interaction_fields:
                fields = interaction.split('_x_')
                if all(field in df.columns for field in fields):
                    # Create the interaction feature
                    X[interaction] = df[fields].astype(str).agg('_'.join, axis=1)
                    if fit:
                        self.encoders[interaction] = CategoricalEncoder()
                        X[interaction] = self.encoders[interaction].fit_transform(X[interaction])
                    else:
                        if interaction in self.encoders:
                            X[interaction] = self.encoders[interaction].transform(X[interaction])
        
        # Ensure all features are numeric
        X = X.astype(float)
        
        # Store feature order during fit
        if fit:
            self.feature_order = list(X.columns)
        
        # Ensure consistent feature order
        if self.feature_order is not None:
            X = X[self.feature_order]
        
        return X
    
    def fit(self, df: pd.DataFrame, target: str, settings: Dict[str, Any]) -> None:
        """Fit model with proper categorical handling and interactions"""
        proxies = settings.get("proxies", [])
        locations = [loc for loc in settings.get("locations", []) 
                    if loc != "___everything___"]
        interactions = settings.get("interactions", [])
        
        self.proxy_fields = proxies
        self.location_fields = locations
        
        # Create feature matrix
        X = self._create_feature_matrix(df, fit=True)
        y = df[target].values
        
        print("\nFitting Random Forest model:")
        print(f"Features being used: {list(X.columns)}")
        
        # Fit model
        self.model.fit(X, y)
        
        # Print feature importances
        importances = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        print("\nFeature importances:")
        for feat, imp in importances.items():
            print(f"--> {feat}: {imp:.4f}")
        
        if self.interaction_fields:
            print("\nInteraction feature importances:")
            interaction_importances = importances[importances.index.isin(self.interaction_fields)]
            for feat, imp in interaction_importances.items():
                print(f"--> {feat}: {imp:.4f}")
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Make predictions with consistent feature handling"""
        # Create feature matrix using same process as fit
        X = self._create_feature_matrix(df, fit=False)
        
        # Verify we have all features
        missing_features = set(self.feature_order) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features during prediction: {missing_features}")
        
        return pd.Series(self.model.predict(X), index=df.index)

    def evaluate(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(df)
        actuals = df[target]
        
        metrics = {
            'mae': np.abs(predictions - actuals).mean(),
            'mape': np.abs((predictions - actuals) / actuals).mean() * 100,
            'rmse': np.sqrt(((predictions - actuals) ** 2).mean()),
            'r2': 1 - ((predictions - actuals) ** 2).sum() / ((actuals - actuals.mean()) ** 2).sum()
        }
        
        return metrics

class LightGBMModel(InferenceModel):
    """LightGBM model with improved validation and parameters"""
    
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.encoders = {}
        self.proxy_fields = None
        self.location_fields = None
        self.interaction_fields = None
        self.imputer = SimpleImputer(strategy='median')
        self.feature_order = None  # Store the order of features from training
    
    def _create_feature_matrix(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Create feature matrix with consistent feature order"""
        # Start with proxy fields
        X = df[self.proxy_fields].copy()
        
        # Handle missing values in numeric variables
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            if fit:
                X[numeric_cols] = self.imputer.fit_transform(X[numeric_cols])
            else:
                X[numeric_cols] = self.imputer.transform(X[numeric_cols])
        
        # Add individual location fields
        for loc in self.location_fields:
            if loc in df.columns:
                if fit:
                    self.encoders[loc] = CategoricalEncoder()
                    X[loc] = self.encoders[loc].fit_transform(df[loc])
                else:
                    if loc in self.encoders:
                        X[loc] = self.encoders[loc].transform(df[loc])
        
        # Add interaction features
        if hasattr(self, 'interaction_fields') and self.interaction_fields:
            for interaction in self.interaction_fields:
                fields = interaction.split('_x_')
                if all(field in df.columns for field in fields):
                    # Create the interaction feature
                    X[interaction] = df[fields].astype(str).agg('_'.join, axis=1)
                    if fit:
                        self.encoders[interaction] = CategoricalEncoder()
                        X[interaction] = self.encoders[interaction].fit_transform(X[interaction])
                    else:
                        if interaction in self.encoders:
                            X[interaction] = self.encoders[interaction].transform(X[interaction])
        
        # Ensure all features are numeric
        X = X.astype(float)
        
        # Store feature order during fit
        if fit:
            self.feature_order = list(X.columns)
        
        # Ensure consistent feature order
        if self.feature_order is not None:
            X = X[self.feature_order]
        
        return X
    
    def fit(self, df: pd.DataFrame, target: str, settings: Dict[str, Any]) -> None:
        """Fit model with proper categorical handling and interactions"""
        proxies = settings.get("proxies", [])
        locations = [loc for loc in settings.get("locations", []) 
                    if loc != "___everything___"]
        interactions = settings.get("interactions", [])
        
        self.proxy_fields = proxies
        self.location_fields = locations
        
        # Create feature matrix
        X = self._create_feature_matrix(df, fit=True)
        y = df[target].values.astype(float)  # Convert to numpy array and ensure float type
        
        print("\nFitting LightGBM model:")
        print(f"Features being used: {list(X.columns)}")
        
        # Fit model
        self.model.fit(X, y)
        
        # Print feature importances
        importances = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        print("\nFeature importances:")
        for feat, imp in importances.items():
            print(f"--> {feat}: {imp:.4f}")
        
        if self.interaction_fields:
            print("\nInteraction feature importances:")
            interaction_importances = importances[importances.index.isin(self.interaction_fields)]
            for feat, imp in interaction_importances.items():
                print(f"--> {feat}: {imp:.4f}")
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Make predictions with consistent feature handling"""
        # Create feature matrix using same process as fit
        X = self._create_feature_matrix(df, fit=False)
        
        # Verify we have all features
        missing_features = set(self.feature_order) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features during prediction: {missing_features}")
        
        return pd.Series(self.model.predict(X), index=df.index)

    def evaluate(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(df)
        actuals = df[target]
        
        metrics = {
            'mae': np.abs(predictions - actuals).mean(),
            'mape': np.abs((predictions - actuals) / actuals).mean() * 100,
            'rmse': np.sqrt(((predictions - actuals) ** 2).mean()),
            'r2': 1 - ((predictions - actuals) ** 2).sum() / ((actuals - actuals.mean()) ** 2).sum()
        }
        
        return metrics

class XGBoostModel(InferenceModel):
    """XGBoost model with improved validation and parameters"""
    
    def __init__(self):
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.encoders = {}
        self.proxy_fields = None
        self.location_fields = None
        self.interaction_fields = None
        self.imputer = SimpleImputer(strategy='median')
        self.feature_order = None  # Store the order of features from training
    
    def _create_feature_matrix(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Create feature matrix with consistent feature order"""
        # Start with proxy fields
        X = df[self.proxy_fields].copy()
        
        # Handle missing values in numeric variables
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            if fit:
                X[numeric_cols] = self.imputer.fit_transform(X[numeric_cols])
            else:
                X[numeric_cols] = self.imputer.transform(X[numeric_cols])
        
        # Add individual location fields
        for loc in self.location_fields:
            if loc in df.columns:
                if fit:
                    self.encoders[loc] = CategoricalEncoder()
                    X[loc] = self.encoders[loc].fit_transform(df[loc])
                else:
                    if loc in self.encoders:
                        X[loc] = self.encoders[loc].transform(df[loc])
        
        # Add interaction features
        if hasattr(self, 'interaction_fields') and self.interaction_fields:
            for interaction in self.interaction_fields:
                fields = interaction.split('_x_')
                if all(field in df.columns for field in fields):
                    # Create the interaction feature
                    X[interaction] = df[fields].astype(str).agg('_'.join, axis=1)
                    if fit:
                        self.encoders[interaction] = CategoricalEncoder()
                        X[interaction] = self.encoders[interaction].fit_transform(X[interaction])
                    else:
                        if interaction in self.encoders:
                            X[interaction] = self.encoders[interaction].transform(X[interaction])
        
        # Ensure all features are numeric
        X = X.astype(float)
        
        # Store feature order during fit
        if fit:
            self.feature_order = list(X.columns)
        
        # Ensure consistent feature order
        if self.feature_order is not None:
            X = X[self.feature_order]
        
        return X
    
    def fit(self, df: pd.DataFrame, target: str, settings: Dict[str, Any]) -> None:
        """Fit model with proper categorical handling and interactions"""
        proxies = settings.get("proxies", [])
        locations = [loc for loc in settings.get("locations", []) 
                    if loc != "___everything___"]
        interactions = settings.get("interactions", [])
        
        self.proxy_fields = proxies
        self.location_fields = locations
        
        # Create feature matrix
        X = self._create_feature_matrix(df, fit=True)
        y = df[target].values.astype(float)  # Convert to numpy array and ensure float type
        
        print("\nFitting XGBoost model:")
        print(f"Features being used: {list(X.columns)}")
        
        # Fit model
        self.model.fit(X, y)
        
        # Print feature importances
        importances = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        print("\nFeature importances:")
        for feat, imp in importances.items():
            print(f"--> {feat}: {imp:.4f}")
        
        if self.interaction_fields:
            print("\nInteraction feature importances:")
            interaction_importances = importances[importances.index.isin(self.interaction_fields)]
            for feat, imp in interaction_importances.items():
                print(f"--> {feat}: {imp:.4f}")
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Make predictions with consistent feature handling"""
        # Create feature matrix using same process as fit
        X = self._create_feature_matrix(df, fit=False)
        
        # Verify we have all features
        missing_features = set(self.feature_order) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features during prediction: {missing_features}")
        
        return pd.Series(self.model.predict(X), index=df.index)

    def evaluate(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(df)
        actuals = df[target]
        
        metrics = {
            'mae': np.abs(predictions - actuals).mean(),
            'mape': np.abs((predictions - actuals) / actuals).mean() * 100,
            'rmse': np.sqrt(((predictions - actuals) ** 2).mean()),
            'r2': 1 - ((predictions - actuals) ** 2).sum() / ((actuals - actuals.mean()) ** 2).sum()
        }
        
        return metrics

class EnsembleModel(InferenceModel):
    """Ensemble model combining LightGBM, XGBoost, and Random Forest"""
    
    def __init__(self):
        self.lgb_model = LightGBMModel()
        self.xgb_model = XGBoostModel()
        self.rf_model = RandomForestModel()
        self.weights = None
        self.encoders = {}
        self.proxy_fields = None
        self.location_fields = None
        self.interaction_fields = None
        self.imputer = SimpleImputer(strategy='median')
        self.feature_order = None
    
    def _create_feature_matrix(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Create feature matrix with consistent feature order"""
        # Start with proxy fields
        X = df[self.proxy_fields].copy()
        
        # Handle missing values in numeric variables
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            if fit:
                X[numeric_cols] = self.imputer.fit_transform(X[numeric_cols])
            else:
                X[numeric_cols] = self.imputer.transform(X[numeric_cols])
        
        # Add individual location fields
        for loc in self.location_fields:
            if loc in df.columns:
                if fit:
                    self.encoders[loc] = CategoricalEncoder()
                    X[loc] = self.encoders[loc].fit_transform(df[loc])
                else:
                    if loc in self.encoders:
                        X[loc] = self.encoders[loc].transform(df[loc])
        
        # Add interaction features
        if hasattr(self, 'interaction_fields') and self.interaction_fields:
            for interaction in self.interaction_fields:
                fields = interaction.split('_x_')
                if all(field in df.columns for field in fields):
                    # Create the interaction feature
                    X[interaction] = df[fields].astype(str).agg('_'.join, axis=1)
                    if fit:
                        self.encoders[interaction] = CategoricalEncoder()
                        X[interaction] = self.encoders[interaction].fit_transform(X[interaction])
                    else:
                        if interaction in self.encoders:
                            X[interaction] = self.encoders[interaction].transform(X[interaction])
        
        # Ensure all features are numeric
        X = X.astype(float)
        
        # Store feature order during fit
        if fit:
            self.feature_order = list(X.columns)
        
        # Ensure consistent feature order
        if self.feature_order is not None:
            X = X[self.feature_order]
        
        return X
    
    def fit(self, df: pd.DataFrame, target: str, settings: Dict[str, Any]) -> None:
        """Fit ensemble model and determine optimal weights"""
        proxies = settings.get("proxies", [])
        locations = [loc for loc in settings.get("locations", []) 
                    if loc != "___everything___"]
        interactions = settings.get("interactions", [])
        
        self.proxy_fields = proxies
        self.location_fields = locations
        
        # Split data for weight optimization
        df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
        
        # Fit individual models
        print("\nFitting LightGBM model...")
        self.lgb_model.fit(df_train, target, settings)
        print("\nFitting XGBoost model...")
        self.xgb_model.fit(df_train, target, settings)
        print("\nFitting Random Forest model...")
        self.rf_model.fit(df_train, target, settings)
        
        # Get predictions on validation set
        lgb_preds = self.lgb_model.predict(df_val)
        xgb_preds = self.xgb_model.predict(df_val)
        rf_preds = self.rf_model.predict(df_val)
        actuals = df_val[target].values.astype(float)  # Convert to numpy array and ensure float type
        
        # Optimize weights to minimize RMSE
        def objective(weights):
            ensemble_preds = (weights[0] * lgb_preds + 
                            weights[1] * xgb_preds + 
                            weights[2] * rf_preds)
            return np.sqrt(((ensemble_preds - actuals) ** 2).mean())
        
        initial_weights = np.array([1/3, 1/3, 1/3])
        bounds = [(0, 1), (0, 1), (0, 1)]
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
        self.weights = result.x
        
        print("\nEnsemble weights:")
        print(f"--> LightGBM: {self.weights[0]:.4f}")
        print(f"--> XGBoost: {self.weights[1]:.4f}")
        print(f"--> Random Forest: {self.weights[2]:.4f}")
        
        # Fit final models on full data
        self.lgb_model.fit(df, target, settings)
        self.xgb_model.fit(df, target, settings)
        self.rf_model.fit(df, target, settings)
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Make predictions using weighted ensemble"""
        lgb_preds = self.lgb_model.predict(df)
        xgb_preds = self.xgb_model.predict(df)
        rf_preds = self.rf_model.predict(df)
        
        return pd.Series(
            self.weights[0] * lgb_preds + 
            self.weights[1] * xgb_preds + 
            self.weights[2] * rf_preds,
            index=df.index
        )
    
    def evaluate(self, df: pd.DataFrame, target: str) -> Dict[str, float]:
        """Evaluate ensemble model performance"""
        predictions = self.predict(df)
        actuals = df[target]
        
        metrics = {
            'mae': np.abs(predictions - actuals).mean(),
            'mape': np.abs((predictions - actuals) / actuals).mean() * 100,
            'rmse': np.sqrt(((predictions - actuals) ** 2).mean()),
            'r2': 1 - ((predictions - actuals) ** 2).sum() / ((actuals - actuals.mean()) ** 2).sum()
        }
        
        return metrics

def get_inference_model(model_type: str) -> InferenceModel:
    """Factory function to get inference model by type"""
    models = {
        'ratio_proxy': RatioProxyModel,
        'smart_ols': SmartOLSModel,
        'random_forest': RandomForestModel,
        'lightgbm': LightGBMModel,
        'xgboost': XGBoostModel,
        'ensemble': EnsembleModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return models[model_type]()

def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard regression metrics handling missing values"""
    # Remove any NaN values from both arrays
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'mape': np.nan
        }
    
    metrics = {
        'mae': mean_absolute_error(y_true_clean, y_pred_clean),
        'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        'r2': r2_score(y_true_clean, y_pred_clean)
    }
    
    # Calculate MAPE safely (avoiding division by zero)
    mask = y_true_clean != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true_clean[mask] - y_pred_clean[mask]) / y_true_clean[mask])) * 100
        metrics['mape'] = mape
    else:
        metrics['mape'] = np.inf
        
    return metrics

def perform_spatial_inference(df: gpd.GeoDataFrame, s_infer: dict, key: str, verbose: bool = False) -> gpd.GeoDataFrame:
    """
    Perform spatial inference using specified model(s)
    
    Args:
        df: Input GeoDataFrame
        s_infer: Inference settings from config
        key: Key field name
        verbose: Whether to print progress
        
    Returns:
        GeoDataFrame with inferred values
    """
    # Suppress all numpy warnings for the entire inference process
    with np.errstate(all='ignore'):
        df_out = df.copy()
        for field in s_infer:
            entry = s_infer[field]
            df_out = _do_perform_spatial_inference(df_out, entry, field, key, verbose=verbose)
        return df_out

def _do_perform_spatial_inference(df: pd.DataFrame, s_infer: dict, field: str, key_field: str, verbose: bool = False) -> pd.DataFrame:
    """Perform spatial inference with validation on filled values"""
    if verbose:
        print(f"\n=== Starting inference for field '{field}' ===")
    
    # Get model settings and create initial masks
    model_settings = s_infer.get("model", {})
    model_type = model_settings.get("type")  # Get model type early
    
    if not model_settings:
        raise ValueError(f"No model settings found for field {field}")
    if not model_type:
        raise ValueError(f"No model type specified for field {field}")
        
    filters = s_infer.get("filters", [])
    fill_fields = s_infer.get("fill", [])
    
    # Create masks properly as boolean Series
    if filters:
        inference_mask = pd.Series(False, index=df.index)
        filter_result = select_filter(df, filters)
        
        if isinstance(filter_result, pd.DataFrame):
            filter_result = filter_result.iloc[:, 0]
            
        if isinstance(filter_result, pd.Series):
            common_indices = df.index.intersection(filter_result.index)
            inference_mask.loc[common_indices] = filter_result.loc[common_indices].astype(bool)
        else:
            if len(filter_result) == len(df):
                inference_mask = pd.Series(filter_result, index=df.index).astype(bool)
            else:
                raise ValueError(f"Filter result length ({len(filter_result)}) does not match DataFrame length ({len(df)})")
        
        training_mask = (~inference_mask) & df[field].notna()
    else:
        inference_mask = pd.Series(df[field].isna(), index=df.index)
        training_mask = pd.Series(df[field].notna(), index=df.index)
    
    if verbose:
        print("\nInitial masks:")
        print(f"--> Total rows: {len(df)}")
        print(f"--> Inference mask True: {inference_mask.sum()}")
        print(f"--> Training mask True: {training_mask.sum()}")
    
    # First identify fill opportunities
    df_result = df.copy()
    fill_validation_data = []
    
    if fill_fields:
        if verbose:
            print(f"\nFilling {field} with known values from: {fill_fields}")
            
        for fill_field in fill_fields:
            if fill_field not in df.columns:
                warnings.warn(f"Fill field '{fill_field}' not found in dataframe")
                continue
            
            # Consider both NA and 0 as missing values
            is_missing = df[field].isna() | df[field].eq(0)
            
            # Create fill mask
            fill_mask = (
                is_missing &
                df[fill_field].notna() & 
                df[fill_field].gt(0)
            )
            
            if fill_mask.sum() > 0:
                # Store validation data
                fill_validation_data.append({
                    'name': fill_field,  # Changed 'field' to 'name'
                    'mask': fill_mask,
                    'true_values': df.loc[fill_mask, fill_field],
                    'rows': df.loc[fill_mask],  # Store full rows for prediction
                    'count': fill_mask.sum()
                })
                
                # Apply fill
                df_result.loc[fill_mask, field] = df.loc[fill_mask, fill_field]
                if verbose:
                    print(f"--> Filled {fill_mask.sum():,} values from {fill_field}")
                
                # Update inference mask
                inference_mask = inference_mask & ~fill_mask
    
    # Prepare training data
    df_train_full = df_result[training_mask].copy()
    df_train_full[field] = pd.to_numeric(df_train_full[field], errors='coerce')
    df_train_full = df_train_full.dropna(subset=[field])
    
    # Split for initial validation
    df_train, df_val = train_test_split(
        df_train_full, 
        test_size=0.1,
        random_state=42
    )
    
    if verbose:
        print("\nData split:")
        print(f"--> Full training samples: {len(df_train_full):,}")
        print(f"--> Training samples: {len(df_train):,}")
        print(f"--> Validation samples: {len(df_val):,}")
    
    # Run experiments if enabled
    if model_settings.get("experiment", False):
        print("\n=== Running Model Experiments ===")
        experiment_models = ['ratio_proxy', 'smart_ols', 'random_forest', 'lightgbm', 'xgboost', 'ensemble']
        best_score = float('-inf')
        best_model_type = None
        validation_results = {}
        fill_validation_results = {}
        
        for exp_type in experiment_models:
            print(f"\nTrying {exp_type} model:")
            try:
                # Create experiment-specific settings
                exp_settings = model_settings.copy()
                if exp_type == 'random_forest' and 'locations' in exp_settings:
                    exp_settings['locations'] = [loc for loc in exp_settings['locations'] 
                                               if loc != '___everything___']
                
                # First do regular validation
                model = get_inference_model(exp_type)
                model.fit(df_train, field, exp_settings)
                
                # Regular validation metrics
                val_predictions = model.predict(df_val)
                train_predictions = model.predict(df_train)
                
                val_metrics = _calculate_metrics(
                    df_val[field].values,
                    val_predictions.values
                )
                train_metrics = _calculate_metrics(
                    df_train[field].values,
                    train_predictions.values
                )
                
                validation_results[exp_type] = {
                    'train': train_metrics,
                    'val': val_metrics
                }
                
                print(f"\nRegular Validation Results:")
                print("\nTraining performance:")
                for metric, value in train_metrics.items():
                    print(f"--> {metric}: {value:.4f}")
                print("\nValidation performance:")
                for metric, value in val_metrics.items():
                    print(f"--> {metric}: {value:.4f}")
                
                # Now validate on filled data
                if fill_validation_data:
                    print("\nValidating on filled data:")
                    
                    # Retrain on full training set
                    model.fit(df_train_full, field, model_settings)
                    
                    fill_results = {}
                    for fill_data in fill_validation_data:
                        fill_name = fill_data['name']
                        rows_to_predict = fill_data['rows']
                        true_values = fill_data['true_values']
                        
                        # Make predictions on filled rows
                        fill_predictions = model.predict(rows_to_predict)
                        
                        # Calculate metrics
                        fill_metrics = _calculate_metrics(
                            true_values.values,
                            fill_predictions.values
                        )
                        
                        fill_results[fill_name] = fill_metrics
                        
                        print(f"\nPerformance on {fill_name} ({fill_data['count']} rows):")
                        for metric, value in fill_metrics.items():
                            print(f"--> {metric}: {value:.4f}")
                    
                    fill_validation_results[exp_type] = fill_results
                
                if val_metrics['r2'] > best_score:
                    best_score = val_metrics['r2']
                    best_model_type = exp_type
                    
            except Exception as e:
                print(f"Error with {exp_type} model: {str(e)}")
                continue  # Skip to next model if there's an error
        
        if best_model_type is not None:
            print(f"\nBest performing model: {best_model_type} (R² = {best_score:.4f})")
            print(f"Currently using: {model_type}")
            
            # Print comparison tables
            print("\nRegular Validation Results:")
            train_df = pd.DataFrame({k: v['train'] for k, v in validation_results.items()}).round(4)
            val_df = pd.DataFrame({k: v['val'] for k, v in validation_results.items()}).round(4)
            print("\nTraining Metrics:")
            print(train_df)
            print("\nValidation Metrics:")
            print(val_df)
            
            if fill_validation_results:
                print("\nFill Validation Results:")
                for fill_data in fill_validation_data:
                    fill_name = fill_data['name']
                    print(f"\n{fill_name} Metrics:")
                    fill_df = pd.DataFrame({
                        k: v[fill_name] for k, v in fill_validation_results.items()
                        if fill_name in v  # Only include models that have results for this fill
                    }).round(4)
                    print(fill_df)
    
    # Fit final model on full training data
    if verbose:
        print("\nFitting final model on full training data...")
    
    model = get_inference_model(model_type)
    model.fit(df_train_full, field, model_settings)
    
    # Make predictions
    predictions = model.predict(df_result[inference_mask])
    
    # Update DataFrame
    df_result.loc[inference_mask, field] = predictions
    df_result[f"inferred_{field}"] = False
    df_result.loc[inference_mask, f"inferred_{field}"] = True
    
    return df_result