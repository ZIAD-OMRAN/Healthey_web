"""
PHASE 2: Neural Network Model — User Preference & Nutrition Prediction
Smart AI Diet & Meal Optimization System
========================================
Architecture:
  1. UserNutritionNet   – predicts personalised daily macro targets from user profile
  2. FoodScoringNet     – scores how well a food matches a user's dietary profile
  3. Collaborative Filter (MF) – learns user-food preference embeddings from feedback
All three models are trained here and saved to models/.
"""

import logging
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1: UserNutritionNet
# Purpose: Given user profile → predict optimal daily calorie & macro targets
# ═══════════════════════════════════════════════════════════════════════════

class UserNutritionNet(nn.Module):
    """
    Deep MLP: user features → [target_kcal, protein_g, fat_g, carbs_g]
    Input: 12 user features (age, weight, height, activity, goal, etc.)
    """
    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU(),   # outputs must be non-negative
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2: FoodScoringNet
# Purpose: Given (user_profile + food_nutrients) → preference score [0,1]
# ═══════════════════════════════════════════════════════════════════════════

class FoodScoringNet(nn.Module):
    """
    Binary-classification MLP that predicts if a user would enjoy a food.
    Inputs: concatenated [user_features, food_nutrient_features]
    Output: sigmoid score (probability of preference)
    """
    def __init__(self, user_dim: int, food_dim: int):
        super().__init__()
        combined = user_dim + food_dim
        self.user_enc = nn.Sequential(
            nn.Linear(user_dim, 64), nn.ReLU(), nn.Dropout(0.2)
        )
        self.food_enc = nn.Sequential(
            nn.Linear(food_dim, 128), nn.ReLU(), nn.Dropout(0.2)
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_x, food_x):
        u = self.user_enc(user_x)
        f = self.food_enc(food_x)
        return self.fusion(torch.cat([u, f], dim=1))


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 3: MatrixFactorisation (Collaborative Filtering)
# Purpose: Embedding-based user-food affinity from feedback ratings
# ═══════════════════════════════════════════════════════════════════════════

class MatrixFactorisation(nn.Module):
    """
    Neural CF with user & item embeddings + bias terms.
    Predicts a rating in [1,5].
    """
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 32):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, emb_dim)
        self.item_emb  = nn.Embedding(n_items, emb_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Initialise
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        combined = torch.cat([u, i], dim=1)
        out = self.mlp(combined) + self.user_bias(user_ids) + self.item_bias(item_ids)
        # Clamp to rating range [1, 5]
        return torch.clamp(out.squeeze(), 1.0, 5.0)


# ═══════════════════════════════════════════════════════════════════════════
# DATA PREPARATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def prepare_user_targets(users: pd.DataFrame):
    """Return (X_scaled, y_scaled, feature_scaler, target_scaler)."""
    goal_le = LabelEncoder()
    act_le  = LabelEncoder()
    diet_le = LabelEncoder()
    gen_le  = LabelEncoder()
    allergy_le = LabelEncoder()

    users = users.copy()
    users["goal_enc"]    = goal_le.fit_transform(users["dietary_goal"])
    users["act_enc"]     = act_le.fit_transform(users["activity_level"])
    users["diet_enc"]    = diet_le.fit_transform(users["diet_type"])
    users["gender_enc"]  = gen_le.fit_transform(users["gender"])
    users["allergy_enc"] = allergy_le.fit_transform(users["food_allergy"])

    feature_cols = [
        "age", "weight_kg", "height_cm", "bmr", "tdee",
        "meals_per_day", "snacks_per_day",
        "goal_enc", "act_enc", "diet_enc", "gender_enc", "allergy_enc",
    ]
    target_cols = ["target_kcal", "target_protein_g", "target_fat_g", "target_carbs_g"]

    X = users[feature_cols].values.astype(np.float32)
    y = users[target_cols].values.astype(np.float32)

    feat_scaler   = StandardScaler()
    target_scaler = StandardScaler()
    X_scaled = feat_scaler.fit_transform(X).astype(np.float32)
    y_scaled = target_scaler.fit_transform(y).astype(np.float32)

    encoders = {
        "goal": goal_le, "activity": act_le, "diet": diet_le,
        "gender": gen_le, "allergy": allergy_le,
        "feature_cols": feature_cols,
    }
    return X_scaled, y_scaled, feat_scaler, target_scaler, encoders


def prepare_scoring_data(users: pd.DataFrame, nutrition: pd.DataFrame, feedback: pd.DataFrame):
    """Build (user_X, food_X, label) triplets from feedback data."""
    # User features
    goal_le = LabelEncoder()
    act_le  = LabelEncoder()
    users = users.copy()
    users["goal_enc"] = goal_le.fit_transform(users["dietary_goal"])
    users["act_enc"]  = act_le.fit_transform(users["activity_level"])
    user_feat_cols = ["age", "weight_kg", "height_cm", "goal_enc", "act_enc",
                      "target_kcal", "target_protein_g"]
    user_lookup = users.set_index("user_id")

    # Food nutrient features (category-level proxy since feedback uses category_id)
    nutrient_cols = [
        "energy_kcal", "protein_g", "fat_g", "carbs_g", "fiber_g",
        "sodium_mg", "calcium_mg", "iron_mg", "is_high_protein",
        "is_low_carb", "is_low_fat", "is_low_calorie", "nutrient_density_score",
    ]
    cat_nutrition = (
        nutrition[[c for c in ["category_encoded"] + nutrient_cols if c in nutrition.columns]]
        .groupby("category_encoded").mean()
        if "category_encoded" in nutrition.columns
        else None
    )

    user_rows, food_rows, labels = [], [], []
    for _, row in feedback.iterrows():
        uid = row["user_id"]
        cat = row["food_category_id"]
        if uid not in user_lookup.index:
            continue
        u = user_lookup.loc[uid]
        user_vec = [u.get(c, 0) for c in user_feat_cols]

        # Simple cat feature vector: category one-hot (28 categories)
        food_vec = [0.0] * 28
        if 1 <= cat <= 28:
            food_vec[cat - 1] = 1.0

        label = float(row["would_eat_again"])
        user_rows.append(user_vec)
        food_rows.append(food_vec)
        labels.append(label)

    X_user = np.array(user_rows, dtype=np.float32)
    X_food = np.array(food_rows, dtype=np.float32)
    y      = np.array(labels, dtype=np.float32)

    u_scaler = StandardScaler()
    X_user   = u_scaler.fit_transform(X_user).astype(np.float32)

    return X_user, X_food, y, u_scaler


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING ROUTINES
# ═══════════════════════════════════════════════════════════════════════════

def train_user_nutrition_net(users: pd.DataFrame) -> UserNutritionNet:
    logger.info("── Training UserNutritionNet ──")
    X, y, feat_scaler, target_scaler, encoders = prepare_user_targets(users)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    tr_ds  = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    tr_dl  = DataLoader(tr_ds, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512)

    model = UserNutritionNet(input_dim=X.shape[1]).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_ctr  = 0
    EPOCHS = 60

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += loss_fn(model(xb), yb).item()
        val_loss /= len(val_dl)
        sched.step(val_loss)

        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch:3d}/{EPOCHS} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODELS_DIR / "user_nutrition_net_best.pt")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 15:
                logger.info(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(MODELS_DIR / "user_nutrition_net_best.pt", weights_only=True))
    joblib.dump(feat_scaler,   MODELS_DIR / "user_feat_scaler.pkl")
    joblib.dump(target_scaler, MODELS_DIR / "user_target_scaler.pkl")
    joblib.dump(encoders,      MODELS_DIR / "user_encoders.pkl")
    torch.save(model.state_dict(), MODELS_DIR / "user_nutrition_net.pt")
    logger.info(f"  ✅ UserNutritionNet saved | best_val_loss={best_val_loss:.4f}")
    return model


def train_food_scoring_net(users: pd.DataFrame, nutrition: pd.DataFrame, feedback: pd.DataFrame) -> FoodScoringNet:
    logger.info("── Training FoodScoringNet ──")
    X_user, X_food, y, u_scaler = prepare_scoring_data(users, nutrition, feedback)

    idx = np.arange(len(y))
    tr_idx, val_idx = train_test_split(idx, test_size=0.15, random_state=42, stratify=y)

    def make_loader(i, shuffle):
        ds = TensorDataset(
            torch.tensor(X_user[i]), torch.tensor(X_food[i]), torch.tensor(y[i])
        )
        return DataLoader(ds, batch_size=512, shuffle=shuffle)

    tr_dl  = make_loader(tr_idx, True)
    val_dl = make_loader(val_idx, False)

    model   = FoodScoringNet(user_dim=X_user.shape[1], food_dim=X_food.shape[1]).to(DEVICE)
    opt     = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    EPOCHS  = 40

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for ub, fb, yb in tr_dl:
            ub, fb, yb = ub.to(DEVICE), fb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(ub, fb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_dl)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for ub, fb, yb in val_dl:
                ub, fb, yb = ub.to(DEVICE), fb.to(DEVICE), yb.to(DEVICE)
                pred = model(ub, fb).squeeze()
                val_loss += loss_fn(pred, yb).item()
                correct  += ((pred > 0.5).float() == yb).sum().item()
        val_loss /= len(val_dl)
        acc = correct / len(val_idx)

        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch:2d}/{EPOCHS} | train={tr_loss:.4f} | val={val_loss:.4f} | acc={acc:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODELS_DIR / "food_scoring_net_best.pt")

    model.load_state_dict(torch.load(MODELS_DIR / "food_scoring_net_best.pt", weights_only=True))
    joblib.dump(u_scaler, MODELS_DIR / "scoring_user_scaler.pkl")
    torch.save(model.state_dict(), MODELS_DIR / "food_scoring_net.pt")
    logger.info(f"  ✅ FoodScoringNet saved | best_val={best_val:.4f}")
    return model


def train_matrix_factorisation(feedback: pd.DataFrame) -> MatrixFactorisation:
    logger.info("── Training MatrixFactorisation (Collaborative Filter) ──")
    fb = feedback.copy()
    u_le = LabelEncoder()
    i_le = LabelEncoder()
    fb["user_idx"] = u_le.fit_transform(fb["user_id"])
    fb["item_idx"] = i_le.fit_transform(fb["food_category_id"])

    n_users = fb["user_idx"].nunique()
    n_items = fb["item_idx"].nunique()

    tr_fb, val_fb = train_test_split(fb, test_size=0.15, random_state=42)

    def make_dl(df, shuffle):
        ds = TensorDataset(
            torch.tensor(df["user_idx"].values, dtype=torch.long),
            torch.tensor(df["item_idx"].values, dtype=torch.long),
            torch.tensor(df["rating"].values,   dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=1024, shuffle=shuffle)

    tr_dl  = make_dl(tr_fb, True)
    val_dl = make_dl(val_fb, False)

    model   = MatrixFactorisation(n_users, n_items, emb_dim=32).to(DEVICE)
    opt     = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    EPOCHS  = 50

    best_rmse = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for u, i, r in tr_dl:
            u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(u, i), r).backward()
            opt.step()

        model.eval()
        preds, reals = [], []
        with torch.no_grad():
            for u, i, r in val_dl:
                u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
                preds.append(model(u, i).cpu())
                reals.append(r.cpu())
        rmse = torch.sqrt(loss_fn(torch.cat(preds), torch.cat(reals))).item()
        if epoch % 10 == 0:
            logger.info(f"  Epoch {epoch:2d}/{EPOCHS} | val_RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), MODELS_DIR / "mf_best.pt")

    model.load_state_dict(torch.load(MODELS_DIR / "mf_best.pt", weights_only=True))
    joblib.dump({"user_le": u_le, "item_le": i_le, "n_users": n_users, "n_items": n_items},
                MODELS_DIR / "mf_encoders.pkl")
    torch.save(model.state_dict(), MODELS_DIR / "mf_model.pt")
    logger.info(f"  ✅ MatrixFactorisation saved | best_val_RMSE={best_rmse:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS (used by later phases)
# ═══════════════════════════════════════════════════════════════════════════

class ModelInference:
    """Convenience wrapper for loading and running all three models."""

    def __init__(self):
        self.feat_scaler   = joblib.load(MODELS_DIR / "user_feat_scaler.pkl")
        self.target_scaler = joblib.load(MODELS_DIR / "user_target_scaler.pkl")
        self.encoders      = joblib.load(MODELS_DIR / "user_encoders.pkl")
        self.mf_encoders   = joblib.load(MODELS_DIR / "mf_encoders.pkl")

        feature_cols = self.encoders["feature_cols"]
        self.unn = UserNutritionNet(input_dim=len(feature_cols)).to(DEVICE)
        self.unn.load_state_dict(torch.load(MODELS_DIR / "user_nutrition_net.pt",
                                            map_location=DEVICE, weights_only=True))
        self.unn.eval()

        self.fsn = FoodScoringNet(user_dim=7, food_dim=28).to(DEVICE)
        self.fsn.load_state_dict(torch.load(MODELS_DIR / "food_scoring_net.pt",
                                            map_location=DEVICE, weights_only=True))
        self.fsn.eval()

        mf_enc = self.mf_encoders
        self.mf = MatrixFactorisation(mf_enc["n_users"], mf_enc["n_items"]).to(DEVICE)
        self.mf.load_state_dict(torch.load(MODELS_DIR / "mf_model.pt",
                                           map_location=DEVICE, weights_only=True))
        self.mf.eval()

    def predict_targets(self, user_dict: dict) -> dict:
        """Predict macro targets from a user profile dict."""
        enc = self.encoders
        feats = np.array([[
            user_dict.get("age", 30),
            user_dict.get("weight_kg", 70),
            user_dict.get("height_cm", 170),
            user_dict.get("bmr", 1600),
            user_dict.get("tdee", 2000),
            user_dict.get("meals_per_day", 3),
            user_dict.get("snacks_per_day", 1),
            enc["goal"].transform([user_dict.get("dietary_goal", "maintenance")])[0],
            enc["activity"].transform([user_dict.get("activity_level", "moderate")])[0],
            enc["diet"].transform([user_dict.get("diet_type", "omnivore")])[0],
            enc["gender"].transform([user_dict.get("gender", "male")])[0],
            enc["allergy"].transform([user_dict.get("food_allergy", "none")])[0],
        ]], dtype=np.float32)
        feats_scaled = self.feat_scaler.transform(feats)
        with torch.no_grad():
            pred = self.unn(torch.tensor(feats_scaled).to(DEVICE)).cpu().numpy()
        raw = self.target_scaler.inverse_transform(pred)[0]
        return {
            "target_kcal":      max(float(raw[0]), 800),
            "target_protein_g": max(float(raw[1]), 30),
            "target_fat_g":     max(float(raw[2]), 20),
            "target_carbs_g":   max(float(raw[3]), 20),
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_model_training():
    logger.info("═" * 60)
    logger.info("PHASE 2 — Neural Network Model Training")
    logger.info("═" * 60)

    users     = pd.read_csv(PROCESSED_DIR / "users.csv")
    nutrition = pd.read_csv(PROCESSED_DIR / "nutrition_full.csv", low_memory=False)
    feedback  = pd.read_csv(PROCESSED_DIR / "feedback.csv")

    unn = train_user_nutrition_net(users)
    fsn = train_food_scoring_net(users, nutrition, feedback)
    mf  = train_matrix_factorisation(feedback)

    logger.info("\n✅ PHASE 2 COMPLETE — All models saved to models/")
    logger.info(f"  UserNutritionNet params: {sum(p.numel() for p in unn.parameters()):,}")
    logger.info(f"  FoodScoringNet params:   {sum(p.numel() for p in fsn.parameters()):,}")
    logger.info(f"  MatrixFact params:       {sum(p.numel() for p in mf.parameters()):,}")
    return unn, fsn, mf


if __name__ == "__main__":
    run_model_training()
