# 🥗 Smart AI Diet & Meal Optimization System

> End-to-end AI-powered diet planning using USDA FoodData Central,
> PyTorch Neural Networks, and Linear Programming optimization.

---

## 📌 Overview

This system ingests **25,760 real foods** from the USDA FoodData Central
database and combines three AI layers to produce fully personalised, 
nutritionally-optimised weekly meal plans:

| Layer | Technology | Purpose |
|---|---|---|
| Data | Pandas + Scikit-learn | Preprocessing, feature engineering |
| Prediction | PyTorch Neural Networks | User macro target prediction |
| Optimization | PuLP (CBC solver) | Nutrient-constrained meal selection |
| Learning | Incremental fine-tuning | Feedback-driven personalisation |
| API | FastAPI | REST interface for all features |

---

## 🗂 Project Structure

```
smart_diet_ai/
├── data/
│   ├── raw/                     ← 24 USDA FoodData Central CSVs
│   └── processed/               ← Cleaned datasets, plans, grocery lists
├── models/
│   ├── neural_networks.py       ← 3 PyTorch models (training + inference)
│   ├── user_nutrition_net.pt    ← Macro target predictor (45k params)
│   ├── food_scoring_net.pt      ← User-food compatibility scorer (37k params)
│   └── mf_model.pt              ← Collaborative filter (169k params)
├── optimization/
│   ├── meal_optimizer.py        ← LP optimization engine (PuLP)
│   └── meal_plan_generator.py   ← Full pipeline integrator
├── utils/
│   ├── data_preprocessing.py    ← Phase 1 preprocessing pipeline
│   ├── grocery_generator.py     ← Grocery list + cost estimation
│   └── feedback_loop.py         ← Online learning from user ratings
├── api/
│   └── main.py                  ← FastAPI REST API (10 endpoints)
├── logs/
├── main.py                      ← Master entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/smart-diet-ai.git
cd smart-diet-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place USDA CSV files in data/raw/
#    (Already included from USDA FoodData Central)
```

---

## 🚀 How to Run

### Run the full pipeline (all phases)
```bash
python main.py pipeline
```

### Generate demo meal plans (3 users)
```bash
python main.py demo
```

### Start the REST API
```bash
python main.py api
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### Run integration tests
```bash
python main.py test
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | System health check |
| `POST` | `/users/register` | Register new user |
| `GET`  | `/users/{id}` | Get user profile + preferences |
| `POST` | `/meal-plan/generate` | Generate 1–14 day meal plan |
| `GET`  | `/meal-plan/{id}` | Get saved meal plan (JSON) |
| `GET`  | `/meal-plan/{id}/report` | Get printable plan (text) |
| `GET`  | `/grocery/{id}` | Get weekly grocery list |
| `GET`  | `/grocery/{id}/text` | Printable grocery list |
| `POST` | `/feedback` | Submit meal rating |
| `POST` | `/feedback/learn` | Trigger model learning cycle |
| `GET`  | `/nutrition/search` | Search foods |
| `GET`  | `/nutrition/{fdc_id}` | Get food nutrient profile |
| `GET`  | `/nutrition/categories/all` | List all food categories |

### Example: Register + Generate Plan

```bash
# Register user
curl -X POST http://localhost:8000/users/register \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1, "name": "Ahmed", "age": 30, "weight_kg": 85.0,
    "height_cm": 178.0, "gender": "male", "activity_level": "moderate",
    "dietary_goal": "weight_loss", "diet_type": "omnivore",
    "food_allergy": "none", "meals_per_day": 3, "snacks_per_day": 1
  }'

# Generate 7-day meal plan
curl -X POST http://localhost:8000/meal-plan/generate \
  -H "Content-Type: application/json" \
  -d '{"user": {...same as above...}, "days": 7}'

# Submit feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "food_category_id": 5, "meal_type": "dinner", "rating": 4.5}'
```

---

## 🧠 AI Architecture

### Model 1 — UserNutritionNet
```
Input (12 features) → Dense(256) → BN → ReLU → Dropout(0.3)
                    → Dense(128) → BN → ReLU → Dropout(0.2)
                    → Dense(64)  → ReLU
                    → Output (4): [kcal, protein_g, fat_g, carbs_g]
```

### Model 2 — FoodScoringNet
```
UserEncoder: Input(7) → Dense(64) → ReLU
FoodEncoder: Input(28) → Dense(128) → ReLU
Fusion: Concat → Dense(128) → BN → ReLU → Dense(64) → Sigmoid
Output: P(user prefers food)
```

### Model 3 — MatrixFactorisation
```
User Embedding (32d) + Item Embedding (32d) → MLP(64→32→1) + Bias
Output: predicted rating [1,5]
```

### Optimization Engine
```
For each meal slot:
  1. Filter foods by diet_type + allergens
  2. Score by nutrient_density_score
  3. Solve 0-1 ILP (PuLP/CBC):
     minimise: Σ weighted slack deviations
     subject to: kcal ≈ slot_target ± 10%
                 protein ≥ slot_target × 85%
                 fat ≤ slot_target × 120%
                 carbs ≤ slot_target × 120%
                 Σ x_i = n_foods
```

---

## 📊 Dataset

**Source**: USDA FoodData Central (Foundation Foods + Market Acquisition)
- 74,175 foods across 28 categories
- 155,243 nutrient measurements
- 23 key nutrients tracked per food
- 10,678 portion size records

**Generated Data**:
- 5,000 synthetic user profiles
- 20,000 synthetic feedback events

---

## 🔬 Supported Dietary Profiles

| Diet Type | Restriction |
|---|---|
| Omnivore | None |
| Vegetarian | No meat/fish |
| Vegan | No animal products |
| Keto | High fat, very low carb |
| Paleo | No grains/legumes/dairy |
| Mediterranean | Emphasis on fish, olive oil, vegetables |

**Dietary Goals**: Weight Loss · Muscle Gain · Maintenance · Heart Health · Diabetic Control

**Allergen Filters**: Gluten · Dairy · Nut · Shellfish

---

## 🔄 Feedback Learning Loop

The system continuously improves through:
1. **Real-time profile update** — every rating immediately updates per-user category preferences
2. **CF model fine-tuning** — mini-batch gradient descent on new ratings (nightly)
3. **FoodScoringNet fine-tuning** — updates the neural scorer with ≥50 new records
4. **Preference persistence** — `user_preferences.json` stores liked/disliked foods per user

---

## 🔧 Git Commands

```bash
# Initial setup
git init
git add .
git commit -m "phase 1: data preprocessing pipeline"
git push origin main

# After each phase
git add .
git commit -m "phase 2: neural network models (UserNutritionNet + CF)"
git push origin main

git add .
git commit -m "phase 3: LP optimization engine (PuLP)"
git push origin main

git add .
git commit -m "phase 4: meal plan generator with NN integration"
git push origin main

git add .
git commit -m "phase 5: grocery list generator with cost estimation"
git push origin main

git add .
git commit -m "phase 6: feedback learning loop (incremental fine-tuning)"
git push origin main

git add .
git commit -m "phase 7: FastAPI REST API with 10 endpoints"
git push origin main

git add .
git commit -m "phase 8: final integration, main.py, README, tests"
git push origin main
```

---

## 🔮 Future Research Improvements

1. **Transformer-based food embedding** (BERT on food descriptions)
2. **Reinforcement Learning meal planner** (reward = long-term health outcomes)
3. **Graph Neural Network** for food ingredient relationships
4. **Multi-objective optimisation** (cost + nutrition + preference simultaneously)
5. **Real user study** to replace synthetic feedback data
6. **Integration with wearable data** (steps, heart rate) for dynamic target adjustment
7. **Recipe generation** using LLM conditioned on selected ingredients
8. **Database backend** (PostgreSQL) replacing in-memory user store
9. **Docker containerisation** for production deployment

---

## 👨‍💻 Authors

Built as a university-level AI research prototype.
Dataset: USDA FoodData Central — https://fdc.nal.usda.gov/
