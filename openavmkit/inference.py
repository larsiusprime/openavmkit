from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import warnings
import geopandas as gpd
from openavmkit.filters import select_filter

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
    """Our current ratio-based proxy approach"""
    
    def __init__(self):
        self.proxy_ratios = {}
        self.proxy_stats = {}
        
    def fit(self, df: pd.DataFrame, target: str, settings: Dict[str, Any]) -> None:
        """
        Fit ratio model using proxy fields
        
        Args:
            df: Training DataFrame
            target: Target field to infer
            settings: Model settings including proxies, locations, group_by
        """
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
            # Calculate ratios first
            valid_mask = (df[target].notna() & 
                        df[proxy].notna() & 
                        df[proxy].gt(0) &
                        df[target].gt(0))
                        
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
            
            # Store statistics
            self.proxy_stats[proxy] = {
                'count': len(df_valid),
                'mean': df_valid[f"ratio_{proxy}"].mean(),
                'std': df_valid[f"ratio_{proxy}"].std(),
                'median': df_valid[f"ratio_{proxy}"].median(),
                'q1': q1,
                'q99': q99
            }
            
            # Calculate global ratio first
            global_ratio = df_valid[f"ratio_{proxy}"].median()
            self.proxy_ratios[(proxy, ())] = global_ratio
            
            # Calculate ratios for each location/group combination
            for location in locations:
                if location == "___everything___":
                    continue
                    
                group_list = group_by.copy() if group_by else []
                group_list.append(location)
                
                # Calculate grouped ratios
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

def get_inference_model(model_type: str) -> InferenceModel:
    """Factory function to get inference model by type"""
    models = {
        'ratio_proxy': RatioProxyModel
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return models[model_type]()

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
    """
    Perform spatial inference using specified model
    """
    if verbose:
        print(f"\n=== Starting inference for field '{field}' ===")
    
    # Get model settings
    model_settings = s_infer.get("model", {})
    if not model_settings:
        raise ValueError(f"No model settings found for field {field}")
        
    model_type = model_settings.get("type")
    if not model_type:
        raise ValueError(f"No model type specified for field {field}")
    
    # Initialize model
    model = get_inference_model(model_type)
    
    # Split data into training and inference sets
    filters = s_infer.get("filters", [])
    
    # Create masks for training and inference
    if filters:
        # Get filter result
        filter_result = select_filter(df, filters)
        
        if isinstance(filter_result, pd.DataFrame):
            # If we got a DataFrame, take the first column
            filter_result = filter_result.iloc[:, 0]
            
        # Create a new boolean mask Series
        inference_mask = pd.Series(False, index=df.index)
        
        if isinstance(filter_result, pd.Series):
            # Find common indices and set them to True
            common_indices = df.index.intersection(filter_result.index)
            # Convert to boolean before assignment
            inference_mask.loc[common_indices] = filter_result.loc[common_indices].astype(bool)
        else:
            # If we got a numpy array or list, validate length
            if len(filter_result) == len(df):
                inference_mask = pd.Series(filter_result, index=df.index, dtype=bool)
            else:
                raise ValueError(f"Filter result length ({len(filter_result)}) does not match DataFrame length ({len(df)})")
        
        # Create training mask (not in inference set and has valid value)
        training_mask = (~inference_mask) & df[field].notna()
    else:
        # If no filters, inference mask is just missing values
        inference_mask = pd.Series(df[field].isna(), index=df.index)
        training_mask = pd.Series(df[field].notna(), index=df.index)
    
    if verbose:
        total_missing = inference_mask.sum()
        print(f"\nInitial state:")
        print(f"--> {total_missing:,} rows need values")

    # First try direct fill from known sources
    fill_fields = s_infer.get("fill", [])
    df_result = df.copy()
    
    if fill_fields:
        if verbose:
            print(f"\nAttempting direct fills...")
            
        for fill_field in fill_fields:
            if fill_field not in df.columns:
                warnings.warn(f"Fill field '{fill_field}' not found in dataframe")
                continue
            
            # Consider both NA and 0 as missing values
            is_missing = df[field].isna() | df[field].eq(0)
            
            # Fill mask
            fill_mask = (
                is_missing &
                df[fill_field].notna() & 
                df[fill_field].gt(0)
            )
            
            if fill_mask.sum() > 0:
                df_result.loc[fill_mask, field] = df.loc[fill_mask, fill_field]
                if verbose:
                    print(f"--> Filled {fill_mask.sum():,} rows from {fill_field}")
                    
                # Update inference mask to exclude filled values
                inference_mask = inference_mask & ~fill_mask
                
        if verbose:
            remaining = inference_mask.sum()
            print(f"--> {remaining:,} rows remaining for inference")

    # Split data for inference
    df_train = df_result[~inference_mask].copy()
    df_to_infer = df_result[inference_mask].copy()
    
    if len(df_train) == 0:
        warnings.warn(f"No training data available for field {field}. Skipping inference.")
        return df_result

    # Fit model
    if verbose:
        print("\nPerforming inference...")
    model.fit(df_train, field, model_settings)
    
    # Make predictions
    predictions = model.predict(df_to_infer)
    
    # Update DataFrame
    df_result.loc[inference_mask, field] = predictions
    
    # Track which values were inferred
    df_result[f"inferred_{field}"] = False
    df_result.loc[inference_mask, f"inferred_{field}"] = True

    # Final statistics
    final_missing = df_result[field].isna().sum()
    if verbose:
        print(f"\nFinal results:")
        print(f"--> {len(predictions):,} values inferred")
        if final_missing > 0:
            print(f"--> {final_missing:,} values remain empty ({final_missing/len(df)*100:.1f}% of total)")

    return df_result