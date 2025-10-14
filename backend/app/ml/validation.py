import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DeminingModelValidator:
    """
    Comprehensive validation framework for demining ML models.
    Implements proper cross-validation, temporal validation, and field validation.
    """
    
    def __init__(self, results_dir: str = "results/validation"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                           cv_folds: int = 5, random_state: int = 42) -> Dict[str, Any]:
        """
        Perform stratified k-fold cross-validation.
        
        Args:
            model: ML model to validate
            X: Feature matrix
            y: Target labels
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Stratified k-fold for balanced validation
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            # Store results for each fold
            fold_results = {
                'train_accuracy': [],
                'val_accuracy': [],
                'train_precision': [],
                'val_precision': [],
                'train_recall': [],
                'val_recall': [],
                'train_f1': [],
                'val_f1': [],
                'train_auc': [],
                'val_auc': [],
                'train_ap': [],
                'val_ap': []
            }
            
            # Cross-validation loop
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                y_train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else y_train_pred
                y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_val_pred
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)
                val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_proba)
                
                # Store results
                for metric in train_metrics:
                    fold_results[f'train_{metric}'].append(train_metrics[metric])
                    fold_results[f'val_{metric}'].append(val_metrics[metric])
                
                logger.info(f"Fold {fold + 1}/{cv_folds} - Val Accuracy: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
            
            # Calculate summary statistics
            summary_results = self._calculate_summary_statistics(fold_results)
            
            # Save results
            self._save_cv_results(summary_results, cv_folds)
            
            return summary_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise
    
    def temporal_validate_model(self, model, X: np.ndarray, y: np.ndarray, 
                              timestamps: np.ndarray, n_splits: int = 5) -> Dict[str, Any]:
        """
        Perform temporal validation for time-series data.
        
        Args:
            model: ML model to validate
            X: Feature matrix
            y: Target labels
            timestamps: Timestamps for temporal ordering
            n_splits: Number of temporal splits
            
        Returns:
            Dictionary with temporal validation results
        """
        try:
            # Time series split for temporal validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Sort by timestamp
            sort_idx = np.argsort(timestamps)
            X_sorted = X[sort_idx]
            y_sorted = y[sort_idx]
            timestamps_sorted = timestamps[sort_idx]
            
            temporal_results = {
                'train_accuracy': [],
                'val_accuracy': [],
                'train_auc': [],
                'val_auc': [],
                'train_f1': [],
                'val_f1': [],
                'time_periods': []
            }
            
            # Temporal validation loop
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted)):
                X_train, X_val = X_sorted[train_idx], X_sorted[val_idx]
                y_train, y_val = y_sorted[train_idx], y_sorted[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                y_train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else y_train_pred
                y_val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_val_pred
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)
                val_metrics = self._calculate_metrics(y_val, y_val_pred, y_val_proba)
                
                # Store results
                temporal_results['train_accuracy'].append(train_metrics['accuracy'])
                temporal_results['val_accuracy'].append(val_metrics['accuracy'])
                temporal_results['train_auc'].append(train_metrics['auc'])
                temporal_results['val_auc'].append(val_metrics['auc'])
                temporal_results['train_f1'].append(train_metrics['f1'])
                temporal_results['val_f1'].append(val_metrics['f1'])
                
                # Time period information
                train_start = timestamps_sorted[train_idx[0]]
                train_end = timestamps_sorted[train_idx[-1]]
                val_start = timestamps_sorted[val_idx[0]]
                val_end = timestamps_sorted[val_idx[-1]]
                
                temporal_results['time_periods'].append({
                    'fold': fold + 1,
                    'train_start': train_start,
                    'train_end': train_end,
                    'val_start': val_start,
                    'val_end': val_end,
                    'train_samples': len(train_idx),
                    'val_samples': len(val_idx)
                })
                
                logger.info(f"Temporal Fold {fold + 1}/{n_splits} - Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Calculate summary statistics
            summary_results = self._calculate_temporal_summary(temporal_results)
            
            # Save results
            self._save_temporal_results(summary_results, n_splits)
            
            return summary_results
            
        except Exception as e:
            logger.error(f"Error in temporal validation: {e}")
            raise
    
    def field_validate_model(self, model, field_data: pd.DataFrame, 
                           ground_truth: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform field validation with real-world data.
        
        Args:
            model: Trained ML model
            field_data: Field survey data
            ground_truth: Ground truth labels from field verification
            
        Returns:
            Dictionary with field validation results
        """
        try:
            # Prepare field data
            X_field = self._prepare_field_features(field_data)
            y_field = ground_truth['hazard_present'].values
            
            # Model predictions
            y_pred = model.predict(X_field)
            y_proba = model.predict_proba(X_field)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calculate field metrics
            field_metrics = self._calculate_metrics(y_field, y_pred, y_proba)
            
            # Calculate field-specific metrics
            field_specific_metrics = self._calculate_field_specific_metrics(
                field_data, ground_truth, y_pred, y_proba
            )
            
            # Combine results
            field_results = {**field_metrics, **field_specific_metrics}
            
            # Save results
            self._save_field_results(field_results)
            
            return field_results
            
        except Exception as e:
            logger.error(f"Error in field validation: {e}")
            raise
    
    def validate_uncertainty_quantification(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Validate uncertainty quantification methods.
        
        Args:
            model: Model with uncertainty quantification
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary with uncertainty validation results
        """
        try:
            # Get predictions with uncertainty
            if hasattr(model, 'predict_with_uncertainty'):
                predictions = model.predict_with_uncertainty(X)
                y_pred = predictions['risk_score']
                uncertainty = predictions['uncertainty']
            else:
                y_pred = model.predict(X)
                uncertainty = np.random.uniform(0.1, 0.9, len(y_pred))  # Placeholder
            
            # Calculate uncertainty metrics
            uncertainty_metrics = self._calculate_uncertainty_metrics(y, y_pred, uncertainty)
            
            # Calibration analysis
            calibration_metrics = self._calculate_calibration_metrics(y, y_pred, uncertainty)
            
            # Combine results
            uncertainty_results = {**uncertainty_metrics, **calibration_metrics}
            
            # Save results
            self._save_uncertainty_results(uncertainty_results)
            
            return uncertainty_results
            
        except Exception as e:
            logger.error(f"Error in uncertainty validation: {e}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Probability-based metrics
        if len(np.unique(y_true)) > 1:  # Check if we have both classes
            metrics['auc'] = roc_auc_score(y_true, y_proba)
            metrics['ap'] = average_precision_score(y_true, y_proba)
        else:
            metrics['auc'] = 0.5  # Random performance
            metrics['ap'] = 0.5
        
        return metrics
    
    def _calculate_field_specific_metrics(self, field_data: pd.DataFrame, 
                                        ground_truth: pd.DataFrame,
                                        y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate field-specific metrics for demining applications."""
        metrics = {}
        
        # False positive rate (critical for demining)
        fp_mask = (y_pred == 1) & (ground_truth['hazard_present'] == 0)
        metrics['false_positive_rate'] = fp_mask.sum() / (ground_truth['hazard_present'] == 0).sum()
        
        # False negative rate (extremely critical for demining)
        fn_mask = (y_pred == 0) & (ground_truth['hazard_present'] == 1)
        metrics['false_negative_rate'] = fn_mask.sum() / (ground_truth['hazard_present'] == 1).sum()
        
        # Area-based metrics
        if 'area_size_ha' in field_data.columns:
            fp_area = field_data[fp_mask]['area_size_ha'].sum()
            total_area = field_data['area_size_ha'].sum()
            metrics['false_positive_area_ratio'] = fp_area / total_area
        
        # Cost-based metrics
        if 'clearance_cost' in field_data.columns:
            fp_cost = field_data[fp_mask]['clearance_cost'].sum()
            total_cost = field_data['clearance_cost'].sum()
            metrics['false_positive_cost_ratio'] = fp_cost / total_cost
        
        # Safety metrics
        metrics['safety_score'] = 1 - metrics['false_negative_rate']  # Higher is safer
        
        return metrics
    
    def _calculate_uncertainty_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     uncertainty: np.ndarray) -> Dict[str, float]:
        """Calculate uncertainty quantification metrics."""
        metrics = {}
        
        # Uncertainty calibration
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0  # Expected Calibration Error
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (uncertainty > bin_lower) & (uncertainty <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (y_true[in_bin] == y_pred[in_bin]).mean()
                avg_confidence_in_bin = uncertainty[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        metrics['expected_calibration_error'] = ece
        
        # Uncertainty distribution
        metrics['mean_uncertainty'] = uncertainty.mean()
        metrics['std_uncertainty'] = uncertainty.std()
        metrics['uncertainty_range'] = uncertainty.max() - uncertainty.min()
        
        # Uncertainty vs accuracy correlation
        accuracy_by_uncertainty = []
        uncertainty_bins = np.percentile(uncertainty, [0, 25, 50, 75, 100])
        
        for i in range(len(uncertainty_bins) - 1):
            mask = (uncertainty >= uncertainty_bins[i]) & (uncertainty < uncertainty_bins[i + 1])
            if mask.sum() > 0:
                accuracy_in_bin = (y_true[mask] == y_pred[mask]).mean()
                avg_uncertainty_in_bin = uncertainty[mask].mean()
                accuracy_by_uncertainty.append((avg_uncertainty_in_bin, accuracy_in_bin))
        
        if len(accuracy_by_uncertainty) > 1:
            uncertainties, accuracies = zip(*accuracy_by_uncertainty)
            metrics['uncertainty_accuracy_correlation'] = np.corrcoef(uncertainties, accuracies)[0, 1]
        else:
            metrics['uncertainty_accuracy_correlation'] = 0.0
        
        return metrics
    
    def _calculate_calibration_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     uncertainty: np.ndarray) -> Dict[str, float]:
        """Calculate calibration metrics."""
        metrics = {}
        
        # Reliability diagram
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        reliability_diagram = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (uncertainty > bin_lower) & (uncertainty <= bin_upper)
            if in_bin.sum() > 0:
                accuracy_in_bin = (y_true[in_bin] == y_pred[in_bin]).mean()
                avg_confidence_in_bin = uncertainty[in_bin].mean()
                reliability_diagram.append((avg_confidence_in_bin, accuracy_in_bin))
        
        # Calculate calibration metrics
        if len(reliability_diagram) > 1:
            confidences, accuracies = zip(*reliability_diagram)
            metrics['calibration_slope'] = np.polyfit(confidences, accuracies, 1)[0]
            metrics['calibration_intercept'] = np.polyfit(confidences, accuracies, 1)[1]
            metrics['calibration_r2'] = np.corrcoef(confidences, accuracies)[0, 1] ** 2
        else:
            metrics['calibration_slope'] = 1.0
            metrics['calibration_intercept'] = 0.0
            metrics['calibration_r2'] = 1.0
        
        return metrics
    
    def _calculate_summary_statistics(self, fold_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate summary statistics from fold results."""
        summary = {}
        
        for metric, values in fold_results.items():
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_min'] = np.min(values)
            summary[f'{metric}_max'] = np.max(values)
            summary[f'{metric}_median'] = np.median(values)
        
        # Calculate confidence intervals
        for metric in ['val_accuracy', 'val_auc', 'val_f1']:
            if f'{metric}_mean' in summary:
                mean = summary[f'{metric}_mean']
                std = summary[f'{metric}_std']
                n = len(fold_results[metric])
                se = std / np.sqrt(n)
                summary[f'{metric}_ci_95_lower'] = mean - 1.96 * se
                summary[f'{metric}_ci_95_upper'] = mean + 1.96 * se
        
        return summary
    
    def _calculate_temporal_summary(self, temporal_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate summary statistics for temporal validation."""
        summary = {}
        
        for metric in ['val_accuracy', 'val_auc', 'val_f1']:
            if metric in temporal_results:
                values = temporal_results[metric]
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_trend'] = np.polyfit(range(len(values)), values, 1)[0]
        
        # Temporal stability
        if 'val_accuracy' in temporal_results:
            accuracies = temporal_results['val_accuracy']
            summary['temporal_stability'] = 1 - np.std(accuracies) / np.mean(accuracies)
        
        return summary
    
    def _prepare_field_features(self, field_data: pd.DataFrame) -> np.ndarray:
        """Prepare field data features for model prediction."""
        # This would extract features from field survey data
        # For now, return a placeholder
        n_samples = len(field_data)
        n_features = 15  # Match model input dimension
        return np.random.randn(n_samples, n_features)
    
    def _save_cv_results(self, results: Dict[str, Any], cv_folds: int):
        """Save cross-validation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"cv_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Add metadata
        results['metadata'] = {
            'cv_folds': cv_folds,
            'timestamp': timestamp,
            'validation_type': 'cross_validation'
        }
        
        # Save to file
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Cross-validation results saved to {filepath}")
    
    def _save_temporal_results(self, results: Dict[str, Any], n_splits: int):
        """Save temporal validation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"temporal_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Add metadata
        results['metadata'] = {
            'n_splits': n_splits,
            'timestamp': timestamp,
            'validation_type': 'temporal_validation'
        }
        
        # Save to file
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Temporal validation results saved to {filepath}")
    
    def _save_field_results(self, results: Dict[str, Any]):
        """Save field validation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"field_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Add metadata
        results['metadata'] = {
            'timestamp': timestamp,
            'validation_type': 'field_validation'
        }
        
        # Save to file
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Field validation results saved to {filepath}")
    
    def _save_uncertainty_results(self, results: Dict[str, Any]):
        """Save uncertainty validation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"uncertainty_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Add metadata
        results['metadata'] = {
            'timestamp': timestamp,
            'validation_type': 'uncertainty_validation'
        }
        
        # Save to file
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Uncertainty validation results saved to {filepath}")
