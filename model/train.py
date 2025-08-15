"""
Tourism Recommendation System - Focused Training Pipeline
========================================================

This module implements focused training for the best performing model:
- Advanced Hybrid Gradient Boosting Model with Enhanced Feature Engineering

Author: Optimized ML Pipeline
Purpose: Production-focused training for tourism-advanced-hybrid-gb model
"""

import os
import logging
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    ndcg_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import clickhouse_connect
from dotenv import load_dotenv
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment Setup
def setup_environment():
    """Setup MLflow and environment configuration."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"MLflow tracking URI set to: {mlflow_uri}")
    
    # Set the experiment name
    mlflow.set_experiment("tourism_recommendation_production")
    
    return mlflow_uri

# Utility Functions
def calculate_ndcg(y_true, y_pred, k=10):
    """Calculate NDCG@K score for ranking evaluation."""
    try:
        y_true_binary = (np.array(y_true) >= 4.0).astype(int)
        return ndcg_score([y_true_binary], [y_pred], k=k)
    except Exception as e:
        logger.warning(f"NDCG calculation failed: {e}")
        return 0.0

# Data Loading
def load_data_from_clickhouse():
    """Load tourism data from ClickHouse database."""
    try:
        clickhouse_host = os.getenv('clickhouse_host')
        clickhouse_port = int(os.getenv('clickhouse_port', 8123))
        clickhouse_user = os.getenv('clickhouse_user', 'default')
        clickhouse_database = os.getenv('clickhouse_database')
        clickhouse_table = os.getenv('clickhouse_table')
        
        if not all([clickhouse_host, clickhouse_database, clickhouse_table]):
            raise ValueError("Missing required ClickHouse configuration")
        
        client = clickhouse_connect.get_client(
            host=clickhouse_host,
            port=clickhouse_port,
            username=clickhouse_user,
            database=clickhouse_database
        )
        
        query = f"SELECT * FROM {clickhouse_table}"
        result = client.query(query)
        columns = result.column_names
        df = pd.DataFrame(result.result_rows, columns=columns)
        client.close()
        
        logger.info(f"Successfully loaded {len(df)} records from ClickHouse")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data from ClickHouse: {e}")
        raise

def train_advanced_hybrid_model():
    """Train advanced hybrid model with enhanced feature engineering."""
    logger.info("Training Advanced Hybrid Model with Enhanced Features...")
    
    df = load_data_from_clickhouse()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    with mlflow.start_run(run_name="advanced_hybrid_model"):
        # Advanced feature engineering
        features = []
        targets = []
        
        # Enhanced statistical features
        user_stats = train_df.groupby('user_id')['user_rating'].agg(['mean', 'std', 'count', 'min', 'max']).fillna(0)
        place_stats = train_df.groupby('place_id')['user_rating'].agg(['mean', 'std', 'count', 'min', 'max']).fillna(0)
        category_stats = train_df.groupby('place_category')['user_rating'].agg(['mean', 'count']).fillna(0)
        city_stats = train_df.groupby('place_city')['user_rating'].agg(['mean', 'count']).fillna(0)
        
        global_mean = train_df['user_rating'].mean()
        global_std = train_df['user_rating'].std()
        
        # User-category and user-city preferences
        user_category_prefs = train_df.groupby(['user_id', 'place_category'])['user_rating'].mean().unstack(fill_value=global_mean)
        user_city_prefs = train_df.groupby(['user_id', 'place_city'])['user_rating'].mean().unstack(fill_value=global_mean)
        
        # Price sensitivity analysis
        user_price_stats = train_df.groupby('user_id')['place_price'].agg(['mean', 'std']).fillna(0)
        
        # Extract features for training
        for _, row in train_df.iterrows():
            user_id = row['user_id']
            place_id = row['place_id']
            category = row['place_category']
            city = row['place_city']
            
            # User features (enhanced)
            user_mean = user_stats.loc[user_id, 'mean'] if user_id in user_stats.index else global_mean
            user_std = user_stats.loc[user_id, 'std'] if user_id in user_stats.index else global_std
            user_count = user_stats.loc[user_id, 'count'] if user_id in user_stats.index else 0
            user_range = user_stats.loc[user_id, 'max'] - user_stats.loc[user_id, 'min'] if user_id in user_stats.index else 0
            
            # Place features (enhanced)
            place_mean = place_stats.loc[place_id, 'mean'] if place_id in place_stats.index else global_mean
            place_std = place_stats.loc[place_id, 'std'] if place_id in place_stats.index else global_std
            place_count = place_stats.loc[place_id, 'count'] if place_id in place_stats.index else 0
            place_popularity = np.log1p(place_count)
            
            # Category and city preferences
            category_mean = category_stats.loc[category, 'mean'] if category in category_stats.index else global_mean
            city_mean = city_stats.loc[city, 'mean'] if city in city_stats.index else global_mean
            
            user_category_pref = user_category_prefs.loc[user_id, category] if user_id in user_category_prefs.index and category in user_category_prefs.columns else global_mean
            user_city_pref = user_city_prefs.loc[user_id, city] if user_id in user_city_prefs.index and city in user_city_prefs.columns else global_mean
            
            # Price features
            place_price = row['place_price']
            user_avg_price = user_price_stats.loc[user_id, 'mean'] if user_id in user_price_stats.index else place_price
            price_ratio = place_price / user_avg_price if user_avg_price > 0 else 1.0
            
            # Contextual features
            place_rating = row['place_average_rating']
            place_duration = row['place_visit_duration_minutes']
            user_age = row['user_age']
            
            # Interaction features
            user_place_deviation = abs(user_mean - place_mean)
            rating_price_ratio = place_rating / np.log1p(place_price) if place_price > 0 else place_rating
            
            feature_vector = [
                # User features
                user_mean, user_std, user_count, user_range,
                # Place features
                place_mean, place_std, place_count, place_popularity,
                # Category/City features
                category_mean, city_mean, user_category_pref, user_city_pref,
                # Price features
                place_price, user_avg_price, price_ratio,
                # Contextual features
                place_rating, place_duration, user_age,
                # Interaction features
                user_place_deviation, rating_price_ratio,
                # Global features
                global_mean, global_std
            ]
            
            features.append(feature_vector)
            targets.append(row['user_rating'])
        
        # Prepare test features with same enhanced feature engineering
        test_features = []
        test_targets = []
        
        for _, row in test_df.iterrows():
            user_id = row['user_id']
            place_id = row['place_id']
            category = row['place_category']
            city = row['place_city']
            
            # Use same feature extraction logic as training
            user_mean = user_stats.loc[user_id, 'mean'] if user_id in user_stats.index else global_mean
            user_std = user_stats.loc[user_id, 'std'] if user_id in user_stats.index else global_std
            user_count = user_stats.loc[user_id, 'count'] if user_id in user_stats.index else 0
            user_range = user_stats.loc[user_id, 'max'] - user_stats.loc[user_id, 'min'] if user_id in user_stats.index else 0
            
            place_mean = place_stats.loc[place_id, 'mean'] if place_id in place_stats.index else global_mean
            place_std = place_stats.loc[place_id, 'std'] if place_id in place_stats.index else global_std
            place_count = place_stats.loc[place_id, 'count'] if place_id in place_stats.index else 0
            place_popularity = np.log1p(place_count)
            
            category_mean = category_stats.loc[category, 'mean'] if category in category_stats.index else global_mean
            city_mean = city_stats.loc[city, 'mean'] if city in city_stats.index else global_mean
            
            user_category_pref = user_category_prefs.loc[user_id, category] if user_id in user_category_prefs.index and category in user_category_prefs.columns else global_mean
            user_city_pref = user_city_prefs.loc[user_id, city] if user_id in user_city_prefs.index and city in user_city_prefs.columns else global_mean
            
            place_price = row['place_price']
            user_avg_price = user_price_stats.loc[user_id, 'mean'] if user_id in user_price_stats.index else place_price
            price_ratio = place_price / user_avg_price if user_avg_price > 0 else 1.0
            
            place_rating = row['place_average_rating']
            place_duration = row['place_visit_duration_minutes']
            user_age = row['user_age']
            
            user_place_deviation = abs(user_mean - place_mean)
            rating_price_ratio = place_rating / np.log1p(place_price) if place_price > 0 else place_rating
            
            feature_vector = [
                user_mean, user_std, user_count, user_range,
                place_mean, place_std, place_count, place_popularity,
                category_mean, city_mean, user_category_pref, user_city_pref,
                place_price, user_avg_price, price_ratio,
                place_rating, place_duration, user_age,
                user_place_deviation, rating_price_ratio,
                global_mean, global_std
            ]
            
            test_features.append(feature_vector)
            test_targets.append(row['user_rating'])
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        test_features_scaled = scaler.transform(test_features)
        
        # Train Gradient Boosting model
        gb_model = GradientBoostingRegressor(
            n_estimators=200, 
            max_depth=8, 
            learning_rate=0.1, 
            subsample=0.8,
            random_state=42
        )
        gb_model.fit(features_scaled, targets)
        
        # Make predictions
        predictions = gb_model.predict(test_features_scaled)
        
        # Evaluate
        mse = mean_squared_error(test_targets, predictions)
        mae = mean_absolute_error(test_targets, predictions)
        rmse = np.sqrt(mse)
        
        # Calculate NDCG
        ndcg_10 = calculate_ndcg(test_targets, predictions, k=10)
        
        # Binary metrics
        threshold = 3.5
        actual_binary = (np.array(test_targets) >= threshold).astype(int)
        pred_binary = (predictions >= threshold).astype(int)
        
        precision = precision_score(actual_binary, pred_binary, average='binary', zero_division=0)
        recall = recall_score(actual_binary, pred_binary, average='binary', zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Feature importance analysis
        feature_names = [
            'user_mean', 'user_std', 'user_count', 'user_range',
            'place_mean', 'place_std', 'place_count', 'place_popularity',
            'category_mean', 'city_mean', 'user_category_pref', 'user_city_pref',
            'place_price', 'user_avg_price', 'price_ratio',
            'place_rating', 'place_duration', 'user_age',
            'user_place_deviation', 'rating_price_ratio',
            'global_mean', 'global_std'
        ]
        
        # Log parameters and metrics
        mlflow.log_params({
            "model_type": "advanced_hybrid_gradient_boosting",
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "n_features": len(features[0]),
            "feature_scaling": "StandardScaler"
        })
        
        mlflow.log_metrics({
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "ndcg_10": ndcg_10,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "feature_importance_mean": np.mean(gb_model.feature_importances_),
            "feature_importance_std": np.std(gb_model.feature_importances_)
        })
        
        # Log top feature importances
        feature_importance_dict = dict(zip(feature_names, gb_model.feature_importances_))
        top_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (feature, importance) in enumerate(top_features):
            mlflow.log_metric(f"top_feature_{i+1}_{feature}", importance)
        
        mlflow.set_tags({
            "model_type": "advanced_hybrid",
            "algorithm": "Gradient Boosting + Enhanced Features",
            "framework": "scikit-learn",
            "feature_engineering": "advanced"
        })
        
        # Register model
        try:
            signature = mlflow.models.infer_signature(features_scaled, predictions)
            mlflow.sklearn.log_model(
                sk_model=gb_model,
                artifact_path="model",
                signature=signature,
                registered_model_name="tourism-advanced-hybrid-gb"
            )
            
            mlflow.sklearn.log_model(
                sk_model=scaler,
                artifact_path="scaler",
                registered_model_name="tourism-advanced-hybrid-scaler"
            )
            logger.info("✅ Advanced hybrid models registered successfully")
        except Exception as e:
            logger.warning(f"Model registration failed: {e}")
        
        logger.info(f"Advanced Hybrid - RMSE: {rmse:.4f}, MAE: {mae:.4f}, F1: {f1:.4f}, NDCG@10: {ndcg_10:.4f}")
        return rmse, mae

if __name__ == "__main__":
    setup_environment()
    train_advanced_hybrid_model()
