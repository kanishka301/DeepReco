import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import sys
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from tensorflow.keras import layers, Model

# 1. App configuration
app = FastAPI(
    title="DeepReco API",
    description="Personalized product recommendations using Neural Collaborative Filtering (NCF).",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
if (BASE_DIR / "Artifacts").exists():
    ARTIFACTS_DIR = BASE_DIR / "Artifacts"
else:
    raise FileNotFoundError(
        "Artifacts folder not found."
    )
# 2. Load saved encoders and supporting artifacts

user_encoder = joblib.load(
    ARTIFACTS_DIR / "user_encoder.pkl"
)

product_encoder = joblib.load(
    ARTIFACTS_DIR / "product_encoder.pkl"
)

sys.modules["numpy._core"] = np.core
sys.modules["numpy._core.numeric"] = np.core.numeric

product_details = pd.read_csv(
    ARTIFACTS_DIR / "product_details.csv"
)

seen = joblib.load(
    ARTIFACTS_DIR / "seen_products.pkl"
)

n_users = len(user_encoder.classes_)
n_items = len(product_encoder.classes_)
embedding_dim = 32

user_input = layers.Input(
    shape=(1,),
    dtype="int32",
    name="user_id",
)

item_input = layers.Input(
    shape=(1,),
    dtype="int32",
    name="product_id",
)

user_embedding = layers.Embedding(
    input_dim=n_users,
    output_dim=embedding_dim,)(user_input)

item_embedding = layers.Embedding(
    input_dim=n_items,
    output_dim=embedding_dim,)(item_input)

user_vec = layers.Flatten()(user_embedding)
item_vec = layers.Flatten()(item_embedding)

x = layers.Concatenate()([user_vec, item_vec])
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation="relu")(x)

output = layers.Dense(1,activation="sigmoid",)(x)

model = Model(
    inputs=[user_input, item_input],
    outputs=output,)

weights_path = ARTIFACTS_DIR / "ncf_weights.npz"

if not weights_path.exists():
    raise FileNotFoundError(
        f"Model weights not found: {weights_path}"
    )

with np.load(weights_path) as weights:
    model.set_weights([
        weights[f"arr_{i}"]
        for i in range(len(weights.files))
    ])

print("NCF model weights loaded successfully!")
print(f"Users: {n_users}")
print(f"Products: {n_items}")

# Map model indices back to original product IDs
product_ids = product_encoder.classes_

# 4. Health check endpoints
@app.get("/")
def home():
    return {
        "message": "DeepReco API is running",
        "model": "Neural Collaborative Filtering (NCF)",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "users": int(n_users),
        "products": int(n_items),
        "model_weights_loaded": True,
    }
# 5. Recommendation endpoint

@app.get("/recommend/{user_id}")
def recommend(
    user_id: str,
    top_k: int = Query(default=10, ge=1, le=50),
):
    # Find the user's encoded index
    user_match = np.where(
        user_encoder.classes_ == user_id
    )[0]

    if len(user_match) == 0:
        raise HTTPException(
            status_code=404,
            detail="User not found in the trained model.",
        )

    user_idx = int(user_match[0])

    # Exclude products already interacted with
    already_seen = seen.get(user_id, set())

    candidate_indices = np.arange(n_items,dtype=np.int32,)

    candidate_ids = product_encoder.inverse_transform(candidate_indices)

    mask = np.array([pid not in already_seen for pid in candidate_ids])

    candidate_indices = candidate_indices[mask]
    candidate_ids = candidate_ids[mask]

    if len(candidate_indices) == 0:
        return {
            "user_id": user_id,
            "count": 0,
            "recommendations": [],
        }

    # Score candidate products for this user
    user_array = np.full(
        len(candidate_indices),
        user_idx,
        dtype=np.int32,)

    scores = model.predict(
        [user_array, candidate_indices],
        batch_size=8192,
        verbose=0,
    ).ravel()

    # Select top-ranked products
    top_indices = np.argsort(-scores)[:top_k]

    results = pd.DataFrame({
        "product_id": candidate_ids[top_indices],
        "NCF_score": scores[top_indices],
    })

    # Attach product metadata
    results = results.merge(
        product_details,
        on="product_id",
        how="left",
    )
    results = results.astype(object).where(
        pd.notna(results),
        None,
    )

    return {
        "user_id": user_id,
        "count": len(results),
        "recommendations": results.to_dict(
            orient="records"
        ),
    }