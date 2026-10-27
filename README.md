# DeepReco - E-Commerce Recommendation System

DeepReco is a deep learning-based e-commerce recommendation system that generates personalized product recommendations using **Neural Collaborative Filtering (NCF)**. It combines TensorFlow/Keras for model inference with FastAPI to expose recommendations through REST API endpoints.

The project demonstrates how a trained recommendation model can be integrated into a backend service and made accessible to an e-commerce frontend.

## Features

- **Personalized recommendations:** Generates product recommendations for a given user ID.
- **Neural Collaborative Filtering:** Learns user and product embeddings to capture interaction patterns.
- **FastAPI backend:** Serves recommendations through HTTP endpoints.
- **Top-K ranking:** Ranks candidate products using the model's predicted interaction scores.
- **Product metadata integration:** Maps recommended product IDs to product information for display.
- **Interactive API documentation:** Enables endpoint testing through Swagger UI.

## Tech Stack

| Component | Technologies |
|---|---|
| Programming language | Python |
| Deep learning | TensorFlow, Keras |
| Recommendation model | Neural Collaborative Filtering (NCF) |
| Backend | FastAPI, Uvicorn |
| Data processing | Pandas, NumPy |
| API testing | Swagger UI |

## How It Works

1. **Data preparation:** User-product interactions are processed and encoded into numerical IDs.
2. **Model inference:** The NCF model uses learned user and product embeddings to estimate interaction scores.
3. **Candidate scoring:** Candidate products are passed through the model to obtain recommendation scores.
4. **Ranking:** Products are ranked by their predicted scores, with previously interacted products filtered according to the implementation.
5. **API response:** FastAPI returns the recommended products so a frontend can display them.

## Model Architecture

The NCF model combines user and product embeddings to learn nonlinear interaction patterns.

The architecture uses:

- User embedding layer
- Product embedding layer
- Embedding concatenation
- Fully connected neural network layers with ReLU activation
- Dropout regularization
- Sigmoid output layer for interaction scoring

The model is trained using binary cross-entropy loss and the Adam optimizer.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/kanishka301/DeepReco.git
cd DeepReco
```

### 2. Create a virtual environment

```bash
python -m venv rec_env
```

Activate it on Windows:

```bash
rec_env\Scripts\activate
```

On macOS or Linux:

```bash
source rec_env/bin/activate
```

### 3. Install dependencies

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

Ensure the installed TensorFlow, NumPy, and other package versions are compatible with the saved model artifacts.

### 4. Start the FastAPI server

From the directory containing `main.py`, run:

```bash
python -m uvicorn main:app --reload
```

The API should be available at:

`http://127.0.0.1:8000`

## API Documentation and Testing

Open the interactive Swagger UI:

`http://127.0.0.1:8000/docs`

Use the available endpoints to test the API and inspect request parameters and response schemas.

### Generate Recommendations

The recommendation endpoint implemented in this project is:

```http
GET /recommend/{user_id}
```

Replace `{user_id}` with a valid user ID supported by the model's user encoder.

Example:

```text
http://127.0.0.1:8000/recommend/USER_ID
```

Use a valid ID from your model's encoded user vocabulary. The exact response fields depend on the API implementation.

## Model Evaluation

The model was evaluated using a sampled offline ranking experiment with 2,000 test cases and 100 negative candidate items per test case.

| Metric | NCF | Popularity Baseline |
|---|---:|---:|
| Recall@10 | 0.7185 | 0.6830 |
| NDCG@10 | 0.4986 | 0.4691 |

- **Recall@10** measures whether the relevant item appears among the top 10 recommendations.
- **NDCG@10** evaluates the ranking quality, giving more credit when relevant items appear closer to the top.

In this evaluation setup, NCF outperformed the popularity baseline on both metrics. These are sampled offline results, not evidence of live production performance.

## Limitations

- Recommendations depend on the users and products represented in the trained model.
- New users and products without learned embeddings require additional handling.
- The model primarily learns from user-product interaction IDs; product metadata is used for display rather than as a direct model input.
- Offline ranking metrics may differ from real-world performance.
- The model does not establish that a user dislikes a product merely because no interaction was recorded.

## Future Improvements

- Add content-based signals such as product category, brand, and description.
- Compare NCF with alternative recommendation architectures.
- Introduce a cold-start strategy for new users and products.
- Evaluate ranking performance under additional sampling strategies.
- Improve recommendation latency and caching for larger candidate sets.
- Expand frontend integration and user-facing recommendation controls.

