import os
import logging
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from typing import List, Dict, Any, Optional
from fastapi.concurrency import run_in_threadpool
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class TourismInput(BaseModel):
    user_id: int
    place_id: int
    place_category: Optional[str] = None
    place_city: Optional[str] = None
    place_price: Optional[float] = None
    place_average_rating: Optional[float] = None
    place_visit_duration_minutes: Optional[int] = None
    user_age: Optional[int] = None


class TourismRecommendation(BaseModel):
    place_id: int
    place_name: Optional[str] = None
    place_category: Optional[str] = None
    place_city: Optional[str] = None
    predicted_rating: float
    confidence_score: Optional[float] = None
    recommendation_reason: Optional[str] = None


app = FastAPI(
    title="Tourism Recommendation API",
    description="An API to provide tourism recommendations using ML models.",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(app)
logging.info("Prometheus instrumentator has been set up.")

# --- Defining Custom Metrics for ML Model ---
recommendations_total = Counter(
    "ml_recommendations_total", "Total number of recommendations served."
)
model_rmse_gauge = Gauge("ml_model_rmse", "Current RMSE of the loaded model.")
prediction_confidence_histogram = Histogram(
    "ml_prediction_confidence", "Distribution of prediction confidence scores."
)
recommendation_diversity_gauge = Gauge(
    "ml_recommendation_diversity", "Diversity score of recommendations."
)

# Environment setup
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

# Model configuration - using the best model from training
MODEL_NAME = "tourism-neural-cf"  # Best model: Neural Collaborative Filtering
MODEL_STAGE = "production"
model = None
scaler = None
model_rmse = None
model_mae = None


@app.on_event("startup")
def load_model():
    """
    Loads the Neural Collaborative Filtering model from the MLflow Model Registry
    during the application's startup.
    """
    global model, model_rmse, model_mae

    try:
        # Load the Neural CF model
        model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"
        logging.info(f"Attempting to load Neural CF model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info(
            f"Neural CF model '{MODEL_NAME}@{MODEL_STAGE}' loaded successfully."
        )

        # Note: Neural CF doesn't need a separate scaler
        logging.info("Neural CF model loaded (no scaler needed for this model type)")

        # Fetch model metrics from MLflow
        logging.info("Fetching model metrics...")
        client = MlflowClient()
        model_version_details = client.get_model_version_by_alias(
            MODEL_NAME, MODEL_STAGE
        )
        run_id = model_version_details.run_id
        logging.info(f"Fetching metrics from Run ID: {run_id}")

        run_data = client.get_run(run_id).data
        model_rmse = run_data.metrics.get("rmse", 0.0)
        model_mae = run_data.metrics.get("mae", 0.0)
        model_ndcg = run_data.metrics.get("ndcg_10", 0.0)
        logging.info(
            f"Model RMSE: {model_rmse}, MAE: {model_mae}, NDCG@10: {model_ndcg}"
        )

        # Set Prometheus metrics
        model_rmse_gauge.set(model_rmse)

    except MlflowException as e:
        model, model_rmse, model_mae = None, 0.0, 0.0
        model_rmse_gauge.set(0.0)
        logging.warning(f"Neural CF model not found in MLflow. Error: {e}")
    except Exception as e:
        model, model_rmse, model_mae = None, 0.0, 0.0
        model_rmse_gauge.set(0.0)
        logging.error(f"Error loading Neural CF model: {e}", exc_info=True)


@app.get("/")
def read_root():
    """
    Root endpoint that provides status information about the API and the loaded model.
    """
    model_status = "ready" if model is not None else "not ready (model not loaded)"
    rmse_info = f"{model_rmse:.4f}" if isinstance(model_rmse, float) else "N/A"
    mae_info = f"{model_mae:.4f}" if isinstance(model_mae, float) else "N/A"

    return {
        "api_status": "ok",
        "api_name": "Tourism Recommendation API",
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_type": "Neural Collaborative Filtering",
        "model_status": model_status,
        "model_rmse": rmse_info,
        "model_mae": mae_info,
        "endpoints": {
            "recommend": "/recommend",
            "batch_recommend": "/batch-recommend",
            "refresh_model": "/refresh-model",
            "model_info": "/model-info",
        },
    }


def create_feature_vector(user_data: Dict[str, Any]) -> List[float]:
    """
    Create feature vector for the model based on user input.
    This matches the feature engineering from train.py
    """
    # Default values (you might want to load these from database)
    global_mean = 3.5
    global_std = 1.0

    # Extract features (matching train.py feature engineering)
    user_id = user_data.get("user_id", 0)
    place_id = user_data.get("place_id", 0)
    place_category = user_data.get("place_category", "Unknown")
    place_city = user_data.get("place_city", "Unknown")
    place_price = user_data.get("place_price", 0.0)
    place_rating = user_data.get("place_average_rating", 3.5)
    place_duration = user_data.get("place_visit_duration_minutes", 120)
    user_age = user_data.get("user_age", 30)

    # Simple feature vector (you can enhance this based on your actual data)
    feature_vector = [
        global_mean,  # user_mean (placeholder)
        global_std,  # user_std (placeholder)
        1,  # user_count (placeholder)
        0,  # user_range (placeholder)
        global_mean,  # place_mean (placeholder)
        global_std,  # place_std (placeholder)
        1,  # place_count (placeholder)
        np.log1p(1),  # place_popularity
        hash(place_category) % 100 / 100,  # category_encoded
        hash(place_city) % 100 / 100,  # city_encoded
        global_mean,  # user_category_pref (placeholder)
        global_mean,  # user_city_pref (placeholder)
        place_price,
        place_price,  # user_avg_price (placeholder)
        1.0,  # price_ratio
        place_rating,
        place_duration,
        user_age,
        0,  # user_place_deviation (placeholder)
        place_rating / np.log1p(max(place_price, 1)),  # rating_price_ratio
        global_mean,
        global_std,
    ]

    return feature_vector


def blocking_recommendation_inference(
    model_instance, scaler_instance, input_features: List[List[float]]
) -> List[Dict[str, Any]]:
    """
    Function to perform recommendation inference on the input features.
    """
    try:
        # Scale features
        if scaler_instance:
            input_features_scaled = scaler_instance.transform(input_features)
        else:
            input_features_scaled = input_features

        # Make predictions
        predictions = model_instance.predict(input_features_scaled)

        # Prepare results
        results = []
        for i, pred in enumerate(predictions):
            confidence_score = (
                0.8  # Placeholder - you can implement actual confidence calculation
            )

            results.append(
                {
                    "place_id": i,  # This should be the actual place_id
                    "predicted_rating": float(pred),
                    "confidence_score": confidence_score,
                    "recommendation_reason": "Based on user preferences and place characteristics",
                }
            )

        return results

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise


@app.post("/recommend", response_model=TourismRecommendation)
async def recommend_single(tourism_input: TourismInput):
    """
    Endpoint untuk memberikan rekomendasi tunggal.
    """
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model is not ready for recommendations."
        )

    try:
        # Create feature vector
        input_dict = tourism_input.dict()
        feature_vector = create_feature_vector(input_dict)

        # Execute recommendation in thread pool
        results = await run_in_threadpool(
            blocking_recommendation_inference, model, scaler, [feature_vector]
        )

        if results:
            result = results[0]
            result["place_id"] = tourism_input.place_id

            # Update metrics
            recommendations_total.inc()
            if result.get("confidence_score"):
                prediction_confidence_histogram.observe(result["confidence_score"])

            return TourismRecommendation(**result)
        else:
            raise HTTPException(status_code=500, detail="No recommendation generated")

    except Exception as e:
        logging.error(f"Error during recommendation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error during recommendation: {str(e)}"
        )


@app.post("/batch-recommend", response_model=List[TourismRecommendation])
async def recommend_batch(tourism_batch: List[TourismInput]):
    """
    Endpoint untuk memberikan rekomendasi batch secara asynchronous.
    """
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model is not ready for recommendations."
        )

    try:
        # Create feature vectors for all inputs
        feature_vectors = []
        for item in tourism_batch:
            input_dict = item.dict()
            feature_vector = create_feature_vector(input_dict)
            feature_vectors.append(feature_vector)

        # Execute batch recommendation in thread pool
        results = await run_in_threadpool(
            blocking_recommendation_inference, model, scaler, feature_vectors
        )

        # Update metrics
        recommendations_total.inc(len(results))

        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            result["place_id"] = tourism_batch[i].place_id
            formatted_results.append(TourismRecommendation(**result))

            if result.get("confidence_score"):
                prediction_confidence_histogram.observe(result["confidence_score"])

        return formatted_results

    except Exception as e:
        logging.error(f"Error during batch recommendation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error during batch recommendation: {str(e)}"
        )


@app.get("/model-info")
def get_model_info():
    """
    Endpoint untuk mendapatkan informasi detail tentang model yang sedang digunakan.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    return {
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_type": "Neural Collaborative Filtering",
        "algorithm": "Deep Learning with Embeddings",
        "framework": "TensorFlow",
        "metrics": {"rmse": model_rmse, "mae": model_mae},
        "architecture": {
            "model_type": "Neural Network",
            "embedding_dim": "Dynamic (sqrt of min(n_users, n_items))",
            "layers": [
                "User & Item Embeddings",
                "Dense 128 (ReLU)",
                "Dropout 0.3",
                "Dense 64 (ReLU)",
                "Dropout 0.3",
                "Dense 32 (ReLU)",
                "Output Layer (Linear)",
            ],
            "regularization": "L2 regularization",
            "optimizer": "Adam (lr=0.001)",
        },
        "features": {
            "input_type": "User-Item pairs",
            "user_features": "User ID (encoded)",
            "item_features": "Item ID (encoded)",
            "collaborative_features": "User-Item interaction patterns",
        },
        "training": {
            "batch_size": 512,
            "epochs": "Up to 50 (with early stopping)",
            "loss_function": "Mean Squared Error (MSE)",
            "evaluation_metrics": [
                "RMSE",
                "MAE",
                "NDCG@10",
                "Precision",
                "Recall",
                "F1",
            ],
        },
        "model_status": "active",
    }


@app.post("/refresh-model")
def refresh_model():
    """
    Endpoint untuk memicu pemuatan ulang model secara manual dari MLflow Model Registry.
    """
    logging.info("Received request to refresh the model.")
    load_model()
    if model:
        return {"message": "Model reloaded successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload the model.")


@app.get("/health")
def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": pd.Timestamp.now().isoformat(),
    }
