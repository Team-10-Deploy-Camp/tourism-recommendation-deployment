"""
Tourism Recommendation System - Enhanced ML Pipeline
===================================================

This module implements multiple recommendation algorithms for tourism data:
- Collaborative Filtering (SVD)
- Content-Based Filtering
- Neural Collaborative Filtering (Deep Learning)
- Hybrid Models with Advanced Feature Engineering
- Ensemble Models

Author: Enhanced ML Pipeline
Purpose: MLOps experiment tracking and model comparison
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
    ndcg_score,
)
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import clickhouse_connect
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Flatten,
    Dense,
    Concatenate,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Environment Setup
def setup_environment():
    """Setup MLflow and environment configuration."""
    # Clear existing AWS env vars and load from .env
    os.environ.pop("AWS_ACCESS_KEY_ID", None)
    os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    os.environ.pop("MLFLOW_S3_ENDPOINT_URL", None)

    # Load environment variables
    load_dotenv(".env")

    # Configure connection to the MLflow Tracking Server
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"MLflow tracking URI set to: {mlflow_uri}")

    # Configure S3/MinIO for artifact storage
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "")

    # Set the experiment name
    mlflow.set_experiment("tourism_recommendation_model")

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


def calculate_diversity(recommendations, item_features=None):
    """Calculate recommendation diversity using cosine similarity."""
    if item_features is None or len(recommendations) < 2:
        return 0.0

    similarities = []
    for i in range(len(recommendations)):
        for j in range(i + 1, len(recommendations)):
            sim = cosine_similarity([item_features[i]], [item_features[j]])[0, 0]
            similarities.append(sim)

    return 1.0 - np.mean(similarities) if similarities else 0.0


def create_user_item_encoders(df):
    """Create label encoders for user and item IDs."""
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df = df.copy()
    df["user_encoded"] = user_encoder.fit_transform(df["user_id"])
    df["item_encoded"] = item_encoder.fit_transform(df["place_id"])

    return df, user_encoder, item_encoder


# Data Loading
def load_data_from_clickhouse():
    """Load tourism data from ClickHouse database."""
    try:
        clickhouse_host = os.getenv("clickhouse_host")
        clickhouse_port = int(os.getenv("clickhouse_port", 8123))
        clickhouse_user = os.getenv("clickhouse_user", "default")
        clickhouse_database = os.getenv("clickhouse_database")
        clickhouse_table = os.getenv("clickhouse_table")

        if not all([clickhouse_host, clickhouse_database, clickhouse_table]):
            raise ValueError("Missing required ClickHouse configuration")

        client = clickhouse_connect.get_client(
            host=clickhouse_host,
            port=clickhouse_port,
            username=clickhouse_user,
            database=clickhouse_database,
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


# Model Training Functions
def train_collaborative_filtering():
    """Train collaborative filtering model using SVD."""
    logger.info("Training Collaborative Filtering (SVD)...")

    df = load_data_from_clickhouse()

    # Create user-item matrix
    user_item_matrix = df.pivot_table(
        index="user_id", columns="place_id", values="user_rating", fill_value=0
    )

    # Create test set
    mask = user_item_matrix.values > 0
    test_indices = np.where(mask)
    n_test = int(0.2 * len(test_indices[0]))
    test_sample = np.random.choice(len(test_indices[0]), n_test, replace=False)
    test_rows = test_indices[0][test_sample]
    test_cols = test_indices[1][test_sample]

    train_matrix = user_item_matrix.copy()
    for i, j in zip(test_rows, test_cols):
        train_matrix.iloc[i, j] = 0

    with mlflow.start_run(run_name="collaborative_filtering_svd"):
        # Train SVD
        svd = TruncatedSVD(n_components=50, random_state=42)
        user_factors = svd.fit_transform(train_matrix)
        item_factors = svd.components_
        predicted_ratings = np.dot(user_factors, item_factors)

        # Evaluate
        test_mask = np.zeros_like(mask)
        test_mask[test_rows, test_cols] = True

        actual_ratings = user_item_matrix.values[test_mask]
        predicted_test = predicted_ratings[test_mask]

        mse = mean_squared_error(actual_ratings, predicted_test)
        mae = mean_absolute_error(actual_ratings, predicted_test)
        rmse = np.sqrt(mse)

        # Log parameters
        mlflow.log_params(
            {
                "model_type": "collaborative_filtering_svd",
                "n_components": 50,
                "n_users": user_item_matrix.shape[0],
                "n_items": user_item_matrix.shape[1],
                "sparsity": 1 - (mask.sum() / mask.size),
            }
        )

        # Log metrics
        mlflow.log_metrics(
            {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "explained_variance": svd.explained_variance_ratio_.sum(),
                "coverage": np.sum(predicted_ratings > 3.0) / predicted_ratings.size,
            }
        )

        # Tag the run
        mlflow.set_tags(
            {
                "model_type": "collaborative_filtering",
                "algorithm": "SVD",
                "framework": "scikit-learn",
            }
        )

        # Register model
        try:
            signature = mlflow.models.infer_signature(
                train_matrix.values, predicted_ratings
            )
            mlflow.sklearn.log_model(
                sk_model=svd,
                artifact_path="model",
                signature=signature,
                registered_model_name="tourism-collaborative-filtering",
            )
            logger.info("✅ Collaborative filtering model registered successfully")
        except Exception as e:
            logger.warning(f"Model registration failed: {e}")

        logger.info(f"Collaborative Filtering - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return rmse, mae


def train_popularity_baseline():
    """Train popularity-based baseline model."""
    logger.info("Training Popularity Baseline...")

    df = load_data_from_clickhouse()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="popularity_baseline"):
        # Calculate average rating per place
        place_avg_ratings = train_df.groupby("place_id")["user_rating"].mean()
        global_avg = train_df["user_rating"].mean()

        # Make predictions
        predictions = []
        actuals = []

        for _, row in test_df.iterrows():
            place_id = row["place_id"]
            actual_rating = row["user_rating"]

            # Use place average, fallback to global average
            pred_rating = place_avg_ratings.get(place_id, global_avg)

            predictions.append(pred_rating)
            actuals.append(actual_rating)

        # Evaluate
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)

        # Log parameters and metrics
        mlflow.log_params(
            {
                "model_type": "popularity_baseline",
                "global_avg_rating": global_avg,
                "n_places": len(place_avg_ratings),
            }
        )

        mlflow.log_metrics({"mse": mse, "mae": mae, "rmse": rmse, "coverage": 1.0})

        mlflow.set_tags(
            {
                "model_type": "baseline",
                "algorithm": "Popularity-based",
                "framework": "simple",
            }
        )

        logger.info(f"Popularity Baseline - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return rmse, mae


def train_content_based():
    """Train content-based filtering model."""
    logger.info("Training Content-Based Filtering...")

    df = load_data_from_clickhouse()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="content_based_tfidf"):
        # Create content features
        place_features = df.groupby("place_id").first().reset_index()

        # Text features
        tfidf = TfidfVectorizer(max_features=100, stop_words="english")
        desc_features = tfidf.fit_transform(
            place_features["place_description"].fillna("")
        )

        # Categorical features
        le_category = LabelEncoder()
        le_city = LabelEncoder()
        place_features["category_encoded"] = le_category.fit_transform(
            place_features["place_category"]
        )
        place_features["city_encoded"] = le_city.fit_transform(
            place_features["place_city"]
        )

        # Numerical features
        scaler = StandardScaler()
        numerical_features = [
            "place_price",
            "place_average_rating",
            "place_visit_duration_minutes",
        ]
        place_features[numerical_features] = scaler.fit_transform(
            place_features[numerical_features]
        )

        # Combine features
        content_matrix = np.hstack(
            [
                desc_features.toarray(),
                place_features[["category_encoded", "city_encoded"]].values,
                place_features[numerical_features].values,
            ]
        )

        # Create user profiles
        user_profiles = {}
        for user_id in train_df["user_id"].unique():
            user_data = train_df[train_df["user_id"] == user_id]
            rated_places = user_data["place_id"].values
            ratings = user_data["user_rating"].values

            place_indices = []
            weighted_ratings = []

            for place_id, rating in zip(rated_places, ratings):
                place_idx = place_features[place_features["place_id"] == place_id].index
                if len(place_idx) > 0:
                    place_indices.append(place_idx[0])
                    weighted_ratings.append(rating)

            if place_indices:
                weighted_ratings = np.array(weighted_ratings)
                place_vectors = content_matrix[place_indices]
                if weighted_ratings.max() > weighted_ratings.min():
                    weights = (weighted_ratings - weighted_ratings.min() + 1) / (
                        weighted_ratings.max() - weighted_ratings.min() + 1
                    )
                    weights = weights / weights.sum()
                    user_profile = np.average(place_vectors, weights=weights, axis=0)
                    user_profiles[user_id] = user_profile

        # Make predictions
        predictions = []
        actuals = []

        for _, row in test_df.iterrows():
            user_id = row["user_id"]
            place_id = row["place_id"]
            actual_rating = row["user_rating"]

            if user_id in user_profiles:
                place_idx = place_features[place_features["place_id"] == place_id].index
                if len(place_idx) > 0:
                    place_idx = place_idx[0]
                    user_profile = user_profiles[user_id].reshape(1, -1)
                    place_vector = content_matrix[place_idx].reshape(1, -1)
                    similarity = cosine_similarity(user_profile, place_vector)[0, 0]
                    predicted_rating = 1 + 4 * (similarity + 1) / 2
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)

        if predictions:
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            rmse = np.sqrt(mse)

            # Binary classification metrics
            threshold = 3.5
            actual_binary = (np.array(actuals) >= threshold).astype(int)
            pred_binary = (np.array(predictions) >= threshold).astype(int)

            precision = precision_score(
                actual_binary, pred_binary, average="binary", zero_division=0
            )
            recall = recall_score(
                actual_binary, pred_binary, average="binary", zero_division=0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            # Log parameters and metrics
            mlflow.log_params(
                {
                    "model_type": "content_based_tfidf",
                    "tfidf_max_features": 100,
                    "n_places": len(place_features),
                    "n_users": df["user_id"].nunique(),
                    "content_features_dim": content_matrix.shape[1],
                }
            )

            mlflow.log_metrics(
                {
                    "mse": mse,
                    "mae": mae,
                    "rmse": rmse,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "coverage": len(user_profiles) / df["user_id"].nunique(),
                }
            )

            mlflow.set_tags(
                {
                    "model_type": "content_based",
                    "algorithm": "TF-IDF + Cosine Similarity",
                    "framework": "scikit-learn",
                }
            )

            # Register models
            try:
                mlflow.sklearn.log_model(
                    sk_model=tfidf,
                    artifact_path="tfidf_vectorizer",
                    registered_model_name="tourism-content-based-tfidf",
                )
                logger.info("✅ Content-based model registered successfully")
            except Exception as e:
                logger.warning(f"Model registration failed: {e}")

            logger.info(
                f"Content-based - RMSE: {rmse:.4f}, MAE: {mae:.4f}, F1: {f1:.4f}"
            )
            return rmse, mae

        return float("inf"), float("inf")


def train_neural_collaborative_filtering():
    """Train Neural Collaborative Filtering model using deep learning."""
    logger.info("Training Neural Collaborative Filtering...")

    df = load_data_from_clickhouse()
    df, _, _ = create_user_item_encoders(df)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    n_users = df["user_encoded"].nunique()
    n_items = df["item_encoded"].nunique()

    with mlflow.start_run(run_name="neural_collaborative_filtering"):
        # Build NCF model
        user_input = Input(shape=(), name="user_input")
        item_input = Input(shape=(), name="item_input")

        # Embedding dimensions
        embedding_dim = min(50, int(np.sqrt(min(n_users, n_items))))

        # User and item embeddings
        user_embedding = Embedding(
            n_users, embedding_dim, embeddings_regularizer=l2(1e-6)
        )(user_input)
        item_embedding = Embedding(
            n_items, embedding_dim, embeddings_regularizer=l2(1e-6)
        )(item_input)

        user_vec = Flatten()(user_embedding)
        item_vec = Flatten()(item_embedding)

        # Concatenate user and item vectors
        concat = Concatenate()([user_vec, item_vec])

        # Deep neural network layers
        dense1 = Dense(128, activation="relu", kernel_regularizer=l2(1e-6))(concat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(64, activation="relu", kernel_regularizer=l2(1e-6))(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        dense3 = Dense(32, activation="relu", kernel_regularizer=l2(1e-6))(dropout2)

        # Output layer
        output = Dense(1, activation="linear")(dense3)

        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

        # Prepare training data
        train_users = train_df["user_encoded"].values
        train_items = train_df["item_encoded"].values
        train_ratings = train_df["user_rating"].values

        test_users = test_df["user_encoded"].values
        test_items = test_df["item_encoded"].values
        test_ratings = test_df["user_rating"].values

        # Train model
        history = model.fit(
            [train_users, train_items],
            train_ratings,
            validation_data=([test_users, test_items], test_ratings),
            epochs=50,
            batch_size=512,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
        )

        # Make predictions
        predictions = model.predict([test_users, test_items]).flatten()

        # Evaluate
        mse = mean_squared_error(test_ratings, predictions)
        mae = mean_absolute_error(test_ratings, predictions)
        rmse = np.sqrt(mse)

        # Calculate NDCG
        ndcg_10 = calculate_ndcg(test_ratings, predictions, k=10)

        # Binary classification metrics
        threshold = 3.5
        actual_binary = (test_ratings >= threshold).astype(int)
        pred_binary = (predictions >= threshold).astype(int)

        precision = precision_score(
            actual_binary, pred_binary, average="binary", zero_division=0
        )
        recall = recall_score(
            actual_binary, pred_binary, average="binary", zero_division=0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Log parameters and metrics
        mlflow.log_params(
            {
                "model_type": "neural_collaborative_filtering",
                "embedding_dim": embedding_dim,
                "n_users": n_users,
                "n_items": n_items,
                "learning_rate": 0.001,
                "batch_size": 512,
                "epochs": len(history.history["loss"]),
            }
        )

        mlflow.log_metrics(
            {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "ndcg_10": ndcg_10,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "final_train_loss": history.history["loss"][-1],
                "final_val_loss": history.history["val_loss"][-1],
            }
        )

        mlflow.set_tags(
            {
                "model_type": "deep_learning",
                "algorithm": "Neural Collaborative Filtering",
                "framework": "tensorflow",
            }
        )

        # Register model
        try:
            mlflow.tensorflow.log_model(
                model=model,
                artifact_path="model",
                registered_model_name="tourism-neural-cf",
            )
            logger.info("✅ Neural CF model registered successfully")
        except Exception as e:
            logger.warning(f"Model registration failed: {e}")

        logger.info(
            f"Neural CF - RMSE: {rmse:.4f}, MAE: {mae:.4f}, NDCG@10: {ndcg_10:.4f}"
        )
        return rmse, mae


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
        user_stats = (
            train_df.groupby("user_id")["user_rating"]
            .agg(["mean", "std", "count", "min", "max"])
            .fillna(0)
        )
        place_stats = (
            train_df.groupby("place_id")["user_rating"]
            .agg(["mean", "std", "count", "min", "max"])
            .fillna(0)
        )
        category_stats = (
            train_df.groupby("place_category")["user_rating"]
            .agg(["mean", "count"])
            .fillna(0)
        )
        city_stats = (
            train_df.groupby("place_city")["user_rating"]
            .agg(["mean", "count"])
            .fillna(0)
        )

        global_mean = train_df["user_rating"].mean()
        global_std = train_df["user_rating"].std()

        # User-category and user-city preferences
        user_category_prefs = (
            train_df.groupby(["user_id", "place_category"])["user_rating"]
            .mean()
            .unstack(fill_value=global_mean)
        )
        user_city_prefs = (
            train_df.groupby(["user_id", "place_city"])["user_rating"]
            .mean()
            .unstack(fill_value=global_mean)
        )

        # Price sensitivity analysis
        user_price_stats = (
            train_df.groupby("user_id")["place_price"].agg(["mean", "std"]).fillna(0)
        )

        # Extract features for training
        for _, row in train_df.iterrows():
            user_id = row["user_id"]
            place_id = row["place_id"]
            category = row["place_category"]
            city = row["place_city"]

            # User features (enhanced)
            user_mean = (
                user_stats.loc[user_id, "mean"]
                if user_id in user_stats.index
                else global_mean
            )
            user_std = (
                user_stats.loc[user_id, "std"]
                if user_id in user_stats.index
                else global_std
            )
            user_count = (
                user_stats.loc[user_id, "count"] if user_id in user_stats.index else 0
            )
            user_range = (
                user_stats.loc[user_id, "max"] - user_stats.loc[user_id, "min"]
                if user_id in user_stats.index
                else 0
            )

            # Place features (enhanced)
            place_mean = (
                place_stats.loc[place_id, "mean"]
                if place_id in place_stats.index
                else global_mean
            )
            place_std = (
                place_stats.loc[place_id, "std"]
                if place_id in place_stats.index
                else global_std
            )
            place_count = (
                place_stats.loc[place_id, "count"]
                if place_id in place_stats.index
                else 0
            )
            place_popularity = np.log1p(place_count)

            # Category and city preferences
            category_mean = (
                category_stats.loc[category, "mean"]
                if category in category_stats.index
                else global_mean
            )
            city_mean = (
                city_stats.loc[city, "mean"]
                if city in city_stats.index
                else global_mean
            )

            user_category_pref = (
                user_category_prefs.loc[user_id, category]
                if user_id in user_category_prefs.index
                and category in user_category_prefs.columns
                else global_mean
            )
            user_city_pref = (
                user_city_prefs.loc[user_id, city]
                if user_id in user_city_prefs.index and city in user_city_prefs.columns
                else global_mean
            )

            # Price features
            place_price = row["place_price"]
            user_avg_price = (
                user_price_stats.loc[user_id, "mean"]
                if user_id in user_price_stats.index
                else place_price
            )
            price_ratio = place_price / user_avg_price if user_avg_price > 0 else 1.0

            # Contextual features
            place_rating = row["place_average_rating"]
            place_duration = row["place_visit_duration_minutes"]
            user_age = row["user_age"]

            # Interaction features
            user_place_deviation = abs(user_mean - place_mean)
            rating_price_ratio = (
                place_rating / np.log1p(place_price)
                if place_price > 0
                else place_rating
            )

            feature_vector = [
                # User features
                user_mean,
                user_std,
                user_count,
                user_range,
                # Place features
                place_mean,
                place_std,
                place_count,
                place_popularity,
                # Category/City features
                category_mean,
                city_mean,
                user_category_pref,
                user_city_pref,
                # Price features
                place_price,
                user_avg_price,
                price_ratio,
                # Contextual features
                place_rating,
                place_duration,
                user_age,
                # Interaction features
                user_place_deviation,
                rating_price_ratio,
                # Global features
                global_mean,
                global_std,
            ]

            features.append(feature_vector)
            targets.append(row["user_rating"])

        # Prepare test features with same enhanced feature engineering
        test_features = []
        test_targets = []

        for _, row in test_df.iterrows():
            user_id = row["user_id"]
            place_id = row["place_id"]
            category = row["place_category"]
            city = row["place_city"]

            # Use same feature extraction logic as training
            user_mean = (
                user_stats.loc[user_id, "mean"]
                if user_id in user_stats.index
                else global_mean
            )
            user_std = (
                user_stats.loc[user_id, "std"]
                if user_id in user_stats.index
                else global_std
            )
            user_count = (
                user_stats.loc[user_id, "count"] if user_id in user_stats.index else 0
            )
            user_range = (
                user_stats.loc[user_id, "max"] - user_stats.loc[user_id, "min"]
                if user_id in user_stats.index
                else 0
            )

            place_mean = (
                place_stats.loc[place_id, "mean"]
                if place_id in place_stats.index
                else global_mean
            )
            place_std = (
                place_stats.loc[place_id, "std"]
                if place_id in place_stats.index
                else global_std
            )
            place_count = (
                place_stats.loc[place_id, "count"]
                if place_id in place_stats.index
                else 0
            )
            place_popularity = np.log1p(place_count)

            category_mean = (
                category_stats.loc[category, "mean"]
                if category in category_stats.index
                else global_mean
            )
            city_mean = (
                city_stats.loc[city, "mean"]
                if city in city_stats.index
                else global_mean
            )

            user_category_pref = (
                user_category_prefs.loc[user_id, category]
                if user_id in user_category_prefs.index
                and category in user_category_prefs.columns
                else global_mean
            )
            user_city_pref = (
                user_city_prefs.loc[user_id, city]
                if user_id in user_city_prefs.index and city in user_city_prefs.columns
                else global_mean
            )

            place_price = row["place_price"]
            user_avg_price = (
                user_price_stats.loc[user_id, "mean"]
                if user_id in user_price_stats.index
                else place_price
            )
            price_ratio = place_price / user_avg_price if user_avg_price > 0 else 1.0

            place_rating = row["place_average_rating"]
            place_duration = row["place_visit_duration_minutes"]
            user_age = row["user_age"]

            user_place_deviation = abs(user_mean - place_mean)
            rating_price_ratio = (
                place_rating / np.log1p(place_price)
                if place_price > 0
                else place_rating
            )

            feature_vector = [
                user_mean,
                user_std,
                user_count,
                user_range,
                place_mean,
                place_std,
                place_count,
                place_popularity,
                category_mean,
                city_mean,
                user_category_pref,
                user_city_pref,
                place_price,
                user_avg_price,
                price_ratio,
                place_rating,
                place_duration,
                user_age,
                user_place_deviation,
                rating_price_ratio,
                global_mean,
                global_std,
            ]

            test_features.append(feature_vector)
            test_targets.append(row["user_rating"])

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
            random_state=42,
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

        precision = precision_score(
            actual_binary, pred_binary, average="binary", zero_division=0
        )
        recall = recall_score(
            actual_binary, pred_binary, average="binary", zero_division=0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Feature importance analysis
        feature_names = [
            "user_mean",
            "user_std",
            "user_count",
            "user_range",
            "place_mean",
            "place_std",
            "place_count",
            "place_popularity",
            "category_mean",
            "city_mean",
            "user_category_pref",
            "user_city_pref",
            "place_price",
            "user_avg_price",
            "price_ratio",
            "place_rating",
            "place_duration",
            "user_age",
            "user_place_deviation",
            "rating_price_ratio",
            "global_mean",
            "global_std",
        ]

        # Log parameters and metrics
        mlflow.log_params(
            {
                "model_type": "advanced_hybrid_gradient_boosting",
                "n_estimators": 200,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "n_features": len(features[0]),
                "feature_scaling": "StandardScaler",
            }
        )

        mlflow.log_metrics(
            {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "ndcg_10": ndcg_10,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "feature_importance_mean": np.mean(gb_model.feature_importances_),
                "feature_importance_std": np.std(gb_model.feature_importances_),
            }
        )

        # Log top feature importances
        feature_importance_dict = dict(
            zip(feature_names, gb_model.feature_importances_)
        )
        top_features = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )[:5]
        for i, (feature, importance) in enumerate(top_features):
            mlflow.log_metric(f"top_feature_{i + 1}_{feature}", importance)

        mlflow.set_tags(
            {
                "model_type": "advanced_hybrid",
                "algorithm": "Gradient Boosting + Enhanced Features",
                "framework": "scikit-learn",
                "feature_engineering": "advanced",
            }
        )

        # Register model
        try:
            signature = mlflow.models.infer_signature(features_scaled, predictions)
            mlflow.sklearn.log_model(
                sk_model=gb_model,
                artifact_path="model",
                signature=signature,
                registered_model_name="tourism-advanced-hybrid-gb",
            )

            mlflow.sklearn.log_model(
                sk_model=scaler,
                artifact_path="scaler",
                registered_model_name="tourism-advanced-hybrid-scaler",
            )
            logger.info("✅ Advanced hybrid models registered successfully")
        except Exception as e:
            logger.warning(f"Model registration failed: {e}")

        logger.info(
            f"Advanced Hybrid - RMSE: {rmse:.4f}, MAE: {mae:.4f}, F1: {f1:.4f}, NDCG@10: {ndcg_10:.4f}"
        )
        return rmse, mae


def train_ensemble_model():
    """Train ensemble model combining multiple approaches."""
    logger.info("Training Ensemble Model...")

    df = load_data_from_clickhouse()
    df, _, _ = create_user_item_encoders(df)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="ensemble_model"):
        # Prepare data for different models
        user_item_matrix = train_df.pivot_table(
            index="user_id", columns="place_id", values="user_rating", fill_value=0
        )

        # Train individual models for ensemble
        logger.info("Training SVD component...")
        svd = TruncatedSVD(n_components=30, random_state=42)
        user_factors = svd.fit_transform(user_item_matrix)
        item_factors = svd.components_

        # Prepare feature-based model
        logger.info("Training feature-based component...")
        features = []
        targets = []

        user_stats = (
            train_df.groupby("user_id")["user_rating"]
            .agg(["mean", "std", "count"])
            .fillna(0)
        )
        place_stats = (
            train_df.groupby("place_id")["user_rating"]
            .agg(["mean", "std", "count"])
            .fillna(0)
        )
        global_mean = train_df["user_rating"].mean()

        for _, row in train_df.iterrows():
            user_id = row["user_id"]
            place_id = row["place_id"]

            user_mean = (
                user_stats.loc[user_id, "mean"]
                if user_id in user_stats.index
                else global_mean
            )
            user_count = (
                user_stats.loc[user_id, "count"] if user_id in user_stats.index else 0
            )
            place_mean = (
                place_stats.loc[place_id, "mean"]
                if place_id in place_stats.index
                else global_mean
            )
            place_count = (
                place_stats.loc[place_id, "count"]
                if place_id in place_stats.index
                else 0
            )

            feature_vector = [
                user_mean,
                user_count,
                place_mean,
                place_count,
                row["place_price"],
                row["place_average_rating"],
                row["place_visit_duration_minutes"],
                row["user_age"],
            ]
            features.append(feature_vector)
            targets.append(row["user_rating"])

        # Train Ridge regression for ensemble
        ridge_model = Ridge(alpha=1.0, random_state=42)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        ridge_model.fit(features_scaled, targets)

        # Make predictions on test set
        ensemble_predictions = []
        test_targets = []

        for _, row in test_df.iterrows():
            user_id = row["user_id"]
            place_id = row["place_id"]
            actual_rating = row["user_rating"]
            test_targets.append(actual_rating)

            predictions_list = []

            # SVD prediction
            if user_id in user_item_matrix.index:
                user_idx = list(user_item_matrix.index).index(user_id)
                if place_id in user_item_matrix.columns:
                    place_idx = list(user_item_matrix.columns).index(place_id)
                    svd_pred = np.dot(
                        user_factors[user_idx], item_factors[:, place_idx]
                    )
                    predictions_list.append(svd_pred)

            # Feature-based prediction
            user_mean = (
                user_stats.loc[user_id, "mean"]
                if user_id in user_stats.index
                else global_mean
            )
            user_count = (
                user_stats.loc[user_id, "count"] if user_id in user_stats.index else 0
            )
            place_mean = (
                place_stats.loc[place_id, "mean"]
                if place_id in place_stats.index
                else global_mean
            )
            place_count = (
                place_stats.loc[place_id, "count"]
                if place_id in place_stats.index
                else 0
            )

            feature_vector = [
                user_mean,
                user_count,
                place_mean,
                place_count,
                row["place_price"],
                row["place_average_rating"],
                row["place_visit_duration_minutes"],
                row["user_age"],
            ]
            feature_scaled = scaler.transform([feature_vector])
            ridge_pred = ridge_model.predict(feature_scaled)[0]
            predictions_list.append(ridge_pred)

            # Simple popularity baseline
            popularity_pred = (
                place_mean if place_id in place_stats.index else global_mean
            )
            predictions_list.append(popularity_pred)

            # Ensemble prediction (weighted average)
            if predictions_list:
                weights = [0.4, 0.4, 0.2]  # SVD, Ridge, Popularity
                weighted_pred = np.average(
                    predictions_list[: len(weights)],
                    weights=weights[: len(predictions_list)],
                )
                ensemble_predictions.append(weighted_pred)
            else:
                ensemble_predictions.append(global_mean)

        # Evaluate ensemble
        mse = mean_squared_error(test_targets, ensemble_predictions)
        mae = mean_absolute_error(test_targets, ensemble_predictions)
        rmse = np.sqrt(mse)

        # Calculate NDCG
        ndcg_10 = calculate_ndcg(test_targets, ensemble_predictions, k=10)

        # Binary metrics
        threshold = 3.5
        actual_binary = (np.array(test_targets) >= threshold).astype(int)
        pred_binary = (np.array(ensemble_predictions) >= threshold).astype(int)

        precision = precision_score(
            actual_binary, pred_binary, average="binary", zero_division=0
        )
        recall = recall_score(
            actual_binary, pred_binary, average="binary", zero_division=0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Log parameters and metrics
        mlflow.log_params(
            {
                "model_type": "ensemble",
                "svd_components": 30,
                "ridge_alpha": 1.0,
                "ensemble_weights": "SVD:0.4, Ridge:0.4, Popularity:0.2",
                "n_base_models": 3,
            }
        )

        mlflow.log_metrics(
            {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "ndcg_10": ndcg_10,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

        mlflow.set_tags(
            {
                "model_type": "ensemble",
                "algorithm": "Weighted Average of SVD + Ridge + Popularity",
                "framework": "scikit-learn",
            }
        )

        # Register models
        try:
            mlflow.sklearn.log_model(
                sk_model=svd,
                artifact_path="svd_model",
                registered_model_name="tourism-ensemble-svd",
            )
            mlflow.sklearn.log_model(
                sk_model=ridge_model,
                artifact_path="ridge_model",
                registered_model_name="tourism-ensemble-ridge",
            )
            logger.info("✅ Ensemble models registered successfully")
        except Exception as e:
            logger.warning(f"Model registration failed: {e}")

        logger.info(
            f"Ensemble Model - RMSE: {rmse:.4f}, MAE: {mae:.4f}, F1: {f1:.4f}, NDCG@10: {ndcg_10:.4f}"
        )
        return rmse, mae


# Main Experiment Function
def run_tourism_recommendation_experiments():
    """Run all models in unified experiment including enhanced models."""
    logger.info("🚀 Starting Enhanced Tourism Recommendation System Experiments...")

    # Setup environment
    mlflow_uri = setup_environment()

    results = {}

    # Define model training functions
    models = {
        "Collaborative Filtering": train_collaborative_filtering,
        "Popularity Baseline": train_popularity_baseline,
        "Content-Based": train_content_based,
        "Neural Collaborative Filtering": train_neural_collaborative_filtering,
        "Advanced Hybrid Model": train_advanced_hybrid_model,
        "Ensemble Model": train_ensemble_model,
    }

    # Train all models
    for model_name, train_func in models.items():
        try:
            logger.info(f"🔄 Running {model_name}...")
            rmse, mae = train_func()
            results[model_name] = {"RMSE": rmse, "MAE": mae}
            logger.info(f"✅ {model_name} completed successfully")
        except Exception as e:
            logger.error(f"❌ {model_name} failed: {e}")
            results[model_name] = {"RMSE": float("inf"), "MAE": float("inf")}

    # Print comprehensive results
    logger.info("\n" + "=" * 80)
    logger.info("🏆 TOURISM RECOMMENDATION SYSTEM - EXPERIMENT RESULTS")
    logger.info("=" * 80)

    successful_models = 0
    for model_name, metrics in results.items():
        rmse = metrics["RMSE"]
        mae = metrics["MAE"]
        status = "✅" if rmse != float("inf") else "❌"
        if rmse != float("inf"):
            successful_models += 1
        logger.info(f"{model_name:35} | {status} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    logger.info("=" * 80)
    logger.info(f"📊 Successful models: {successful_models}/{len(results)}")
    logger.info(f"🎯 Experiment: tourism_recommendation_enhanced")
    logger.info(f"🔗 MLflow URI: {mlflow_uri}")

    # Find best model
    valid_results = {k: v for k, v in results.items() if v["RMSE"] != float("inf")}
    if valid_results:
        best_model = min(valid_results.keys(), key=lambda x: valid_results[x]["RMSE"])
        logger.info(
            f"\n🏆 Best Model: {best_model} (RMSE: {valid_results[best_model]['RMSE']:.4f})"
        )

        # Show improvement over baseline
        if "Popularity Baseline" in valid_results:
            baseline_rmse = valid_results["Popularity Baseline"]["RMSE"]
            best_rmse = valid_results[best_model]["RMSE"]
            improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100
            logger.info(f"📈 Improvement over baseline: {improvement:.2f}%")

    # Enhanced models performance summary
    enhanced_models = [
        "Neural Collaborative Filtering",
        "Advanced Hybrid Model",
        "Ensemble Model",
    ]
    enhanced_working = sum(1 for model in enhanced_models if model in valid_results)
    logger.info(
        f"\n🚀 Enhanced models working: {enhanced_working}/{len(enhanced_models)}"
    )

    if enhanced_working > 0:
        logger.info("✨ Successfully enhanced the recommendation system with:")
        for model in enhanced_models:
            if model in valid_results:
                rmse = valid_results[model]["RMSE"]
                logger.info(f"   • {model}: RMSE {rmse:.4f}")

    logger.info(f"\n📊 Total experiment models: {len(results)}")
    logger.info(f"🎯 MLOps tracking: All experiments logged to MLflow")
    logger.info("🎉 Tourism Recommendation System Experiments Complete!")

    return results


if __name__ == "__main__":
    run_tourism_recommendation_experiments()
