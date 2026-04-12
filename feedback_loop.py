"""
PHASE 6: Feedback Learning Loop
Smart AI Diet & Meal Optimization System
==========================================
Implements an online-learning feedback system that:
  1. Collects new user meal ratings
  2. Updates the MatrixFactorisation (CF) model incrementally
  3. Fine-tunes the FoodScoringNet on accumulated feedback
  4. Adjusts per-user food affinity scores stored in a preference profile
  5. Persists everything so each re-run builds on prior knowledge
"""

import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
FEEDBACK_LOG  = PROCESSED_DIR / "feedback_log.jsonl"
PREFERENCE_DB = PROCESSED_DIR / "user_preferences.json"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# FEEDBACK STORE
# ===========================================================================

class FeedbackStore:
    """
    Append-only JSONL log of user feedback events.
    Loads history and supports efficient batch retrieval.
    """

    def __init__(self, path: Path = FEEDBACK_LOG):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def log(self, event: dict):
        """Append a single feedback event."""
        event["timestamp"] = datetime.utcnow().isoformat()
        with open(self.path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def load_all(self) -> pd.DataFrame:
        """Load full feedback history as DataFrame."""
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if not records:
            return pd.DataFrame(columns=[
                "user_id", "fdc_id", "food_category_id",
                "meal_type", "rating", "would_eat_again",
                "portion_satisfied", "timestamp",
            ])
        return pd.DataFrame(records)

    def load_since(self, since_iso: str) -> pd.DataFrame:
        df = self.load_all()
        if df.empty:
            return df
        return df[df["timestamp"] >= since_iso]

    def count(self) -> int:
        try:
            with open(self.path) as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0


# ===========================================================================
# USER PREFERENCE PROFILE
# ===========================================================================

class UserPreferenceProfile:
    """
    Lightweight per-user preference store.
    Maintains:
      - avg rating per food category
      - liked / disliked food fdc_ids
      - taste vector (numeric summary)
    """

    def __init__(self, path: Path = PREFERENCE_DB):
        self.path = path
        self._data: dict = {}
        if path.exists():
            with open(path) as f:
                self._data = json.load(f)

    def update(self, user_id: int, feedback_batch: pd.DataFrame):
        """Update a user's preference profile from a batch of feedback events."""
        uid = str(user_id)
        if uid not in self._data:
            self._data[uid] = {
                "category_ratings": {},
                "liked_fdc_ids":    [],
                "disliked_fdc_ids": [],
                "total_feedback":   0,
                "avg_rating":       0.0,
            }

        prof = self._data[uid]
        for _, row in feedback_batch.iterrows():
            cat  = str(row.get("food_category_id", "?"))
            rat  = float(row.get("rating", 3.0))
            fdc  = row.get("fdc_id", None)
            eat_raw = row.get("would_eat_again", None)
            eat  = int(eat_raw) if eat_raw is not None else int(rat >= 3.5)

            # Update category running average
            if cat not in prof["category_ratings"]:
                prof["category_ratings"][cat] = {"sum": 0.0, "count": 0}
            prof["category_ratings"][cat]["sum"]   += rat
            prof["category_ratings"][cat]["count"] += 1

            # Track liked / disliked foods
            if fdc and eat == 1 and str(fdc) not in prof["liked_fdc_ids"]:
                prof["liked_fdc_ids"].append(str(fdc))
            if fdc and eat == 0 and str(fdc) not in prof["disliked_fdc_ids"]:
                prof["disliked_fdc_ids"].append(str(fdc))

        # Overall averages
        all_ratings = feedback_batch["rating"].dropna().tolist()
        if all_ratings:
            n_prev = prof["total_feedback"]
            old_avg = prof["avg_rating"]
            new_n   = n_prev + len(all_ratings)
            prof["avg_rating"]     = (old_avg * n_prev + sum(all_ratings)) / new_n
            prof["total_feedback"] = new_n

        self._data[uid] = prof

    def get(self, user_id: int) -> dict:
        return self._data.get(str(user_id), {})

    def get_preferred_categories(self, user_id: int, top_n: int = 5) -> list[str]:
        prof = self.get(user_id)
        cat_avgs = {
            cat: v["sum"] / v["count"]
            for cat, v in prof.get("category_ratings", {}).items()
            if v["count"] > 0
        }
        return sorted(cat_avgs, key=cat_avgs.get, reverse=True)[:top_n]

    def get_disliked_foods(self, user_id: int) -> list[str]:
        return self.get(user_id).get("disliked_fdc_ids", [])

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)
        logger.info(f"Preference profiles saved -> {self.path}")


# ===========================================================================
# INCREMENTAL CF MODEL UPDATER
# ===========================================================================

class CFModelUpdater:
    """
    Performs mini-batch fine-tuning of MatrixFactorisation on new feedback.
    Uses the existing user/item encoders; gracefully handles unseen IDs.
    """

    def __init__(self):
        from models.neural_networks import MatrixFactorisation
        self.MF_class = MatrixFactorisation
        self._load()

    def _load(self):
        enc = joblib.load(MODELS_DIR / "mf_encoders.pkl")
        self.user_le: LabelEncoder = enc["user_le"]
        self.item_le: LabelEncoder = enc["item_le"]
        self.n_users = enc["n_users"]
        self.n_items = enc["n_items"]
        self.model = self.MF_class(self.n_users, self.n_items, emb_dim=32).to(DEVICE)
        self.model.load_state_dict(
            torch.load(MODELS_DIR / "mf_model.pt", map_location=DEVICE, weights_only=True)
        )

    def _encode_safe(self, le: LabelEncoder, values, fallback: int = 0) -> np.ndarray:
        """Encode with fallback for unseen labels."""
        out = []
        for v in values:
            try:
                out.append(le.transform([v])[0])
            except ValueError:
                out.append(fallback)
        return np.array(out, dtype=np.int64)

    def update(self, new_feedback: pd.DataFrame, epochs: int = 5, lr: float = 1e-4):
        if new_feedback.empty:
            logger.info("  No new feedback to update CF model.")
            return

        user_idx = self._encode_safe(self.user_le, new_feedback["user_id"].values)
        item_idx = self._encode_safe(self.item_le, new_feedback["food_category_id"].values)
        ratings  = new_feedback["rating"].fillna(3.0).values.astype(np.float32)

        ds = TensorDataset(
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(item_idx, dtype=torch.long),
            torch.tensor(ratings,  dtype=torch.float32),
        )
        dl      = DataLoader(ds, batch_size=min(256, len(ds)), shuffle=True)
        opt     = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(1, epochs + 1):
            ep_loss = 0.0
            for u, i, r in dl:
                u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(self.model(u, i), r)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            if epoch == epochs:
                logger.info(f"  CF fine-tune | epoch {epoch}/{epochs} | loss={ep_loss/len(dl):.4f}")

        torch.save(self.model.state_dict(), MODELS_DIR / "mf_model.pt")
        logger.info(f"  CF model updated with {len(new_feedback)} new feedback records.")

    def predict_affinity(self, user_id: int, category_ids: list[int]) -> dict[int, float]:
        """Predict affinity scores (1-5) for a list of category ids for one user."""
        self.model.eval()
        uid_enc = self._encode_safe(self.user_le, [user_id] * len(category_ids))
        cat_enc = self._encode_safe(self.item_le, category_ids)
        with torch.no_grad():
            scores = self.model(
                torch.tensor(uid_enc, dtype=torch.long).to(DEVICE),
                torch.tensor(cat_enc, dtype=torch.long).to(DEVICE),
            ).cpu().numpy()
        return {cat: float(scores[i]) for i, cat in enumerate(category_ids)}


# ===========================================================================
# FOOD SCORING NET FINE-TUNER
# ===========================================================================

class ScoringNetFineTuner:
    """Fine-tunes FoodScoringNet on accumulated feedback."""

    def __init__(self):
        from models.neural_networks import FoodScoringNet
        self.model = FoodScoringNet(user_dim=7, food_dim=28).to(DEVICE)
        self.model.load_state_dict(
            torch.load(MODELS_DIR / "food_scoring_net.pt", map_location=DEVICE, weights_only=True)
        )
        self.u_scaler = joblib.load(MODELS_DIR / "scoring_user_scaler.pkl")
        self.users    = pd.read_csv(PROCESSED_DIR / "users.csv")

    def fine_tune(self, feedback: pd.DataFrame, epochs: int = 3, lr: float = 5e-5):
        if len(feedback) < 10:
            logger.info("  Not enough data for FoodScoringNet fine-tune (need ?10).")
            return

        user_lookup = self.users.set_index("user_id")
        user_feat_cols = ["age", "weight_kg", "height_cm", "target_kcal", "target_protein_g"]
        # Pad to 7 features
        user_feat_cols = (user_feat_cols + ["meals_per_day", "snacks_per_day"])[:7]

        X_user, X_food, y_labels = [], [], []
        for _, row in feedback.iterrows():
            uid = row["user_id"]
            if uid not in user_lookup.index:
                continue
            u = user_lookup.loc[uid]
            u_vec = [float(u.get(c, 0)) for c in user_feat_cols]
            cat   = int(row.get("food_category_id", 1))
            f_vec = [0.0] * 28
            if 1 <= cat <= 28:
                f_vec[cat - 1] = 1.0
            X_user.append(u_vec)
            X_food.append(f_vec)
            y_labels.append(float(row.get("would_eat_again", int(row["rating"] >= 3.5))))

        if len(X_user) < 10:
            logger.info("  Insufficient matched users for fine-tune.")
            return

        X_u = self.u_scaler.transform(np.array(X_user, dtype=np.float32))
        X_f = np.array(X_food, dtype=np.float32)
        y   = np.array(y_labels, dtype=np.float32)

        ds  = TensorDataset(torch.tensor(X_u), torch.tensor(X_f), torch.tensor(y))
        dl  = DataLoader(ds, batch_size=128, shuffle=True)
        opt = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        self.model.train()
        for epoch in range(1, epochs + 1):
            ep_loss = 0.0
            for ub, fb, yb in dl:
                ub, fb, yb = ub.to(DEVICE), fb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = loss_fn(self.model(ub, fb).squeeze(), yb)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            if epoch == epochs:
                logger.info(f"  ScoringNet fine-tune | epoch {epoch}/{epochs} | loss={ep_loss/len(dl):.4f}")

        torch.save(self.model.state_dict(), MODELS_DIR / "food_scoring_net.pt")
        logger.info(f"  FoodScoringNet updated with {len(X_u)} samples.")


# ===========================================================================
# FEEDBACK LEARNING ORCHESTRATOR
# ===========================================================================

class FeedbackLearningLoop:
    """
    Main orchestrator.
    Call  .submit_feedback()  for single events (real-time).
    Call  .run_learning_cycle()  periodically (batch update).
    """

    def __init__(self):
        self.store      = FeedbackStore()
        self.preferences = UserPreferenceProfile()
        self.cf_updater  = CFModelUpdater()
        self.scorer_ft   = ScoringNetFineTuner()
        logger.info(f"FeedbackLearningLoop ready | existing records: {self.store.count()}")

    def submit_feedback(self, event: dict):
        """
        Accept a single feedback event from a user.
        event keys: user_id, fdc_id, food_category_id, meal_type, rating,
                    would_eat_again (optional), portion_satisfied (optional)
        """
        if "would_eat_again" not in event:
            event["would_eat_again"] = int(event.get("rating", 3.0) >= 3.5)
        if "portion_satisfied" not in event:
            event["portion_satisfied"] = 1
        self.store.log(event)

        # Immediately update preference profile
        row_df = pd.DataFrame([event])
        self.preferences.update(event["user_id"], row_df)
        self.preferences.save()
        logger.info(f"  Feedback logged: user={event['user_id']} | "
                    f"cat={event.get('food_category_id')} | rating={event.get('rating')}")

    def run_learning_cycle(self, min_new_records: int = 50):
        """
        Batch learning cycle: fine-tune models on all accumulated feedback.
        Designed to run nightly / periodically.
        """
        logger.info("-- Running Feedback Learning Cycle --")
        all_fb = self.store.load_all()

        if all_fb.empty:
            logger.info("  No feedback records found.")
            return

        logger.info(f"  Total feedback records: {len(all_fb)}")

        # Update CF model
        self.cf_updater.update(all_fb, epochs=5, lr=1e-4)

        # Fine-tune scoring net
        if len(all_fb) >= min_new_records:
            self.scorer_ft.fine_tune(all_fb, epochs=3, lr=5e-5)

        # Rebuild preference profiles for all users
        for uid, grp in all_fb.groupby("user_id"):
            self.preferences.update(uid, grp)
        self.preferences.save()

        logger.info(f"  Preference profiles updated for {all_fb['user_id'].nunique()} users.")
        logger.info("[OK] Feedback learning cycle complete.")

    def get_user_preferences(self, user_id: int) -> dict:
        return self.preferences.get(user_id)

    def get_affinity_scores(self, user_id: int) -> dict:
        return self.cf_updater.predict_affinity(user_id, list(range(1, 29)))


# ===========================================================================
# MAIN
# ===========================================================================

def run_feedback_demo():
    logger.info("=" * 60)
    logger.info("PHASE 6 - Feedback Learning Loop")
    logger.info("=" * 60)

    loop = FeedbackLearningLoop()

    # -- Simulate new user feedback events --------------------------------
    demo_events = [
        {"user_id": 1, "fdc_id": 319877, "food_category_id": 13, "meal_type": "dinner",
         "rating": 4.5, "would_eat_again": 1},
        {"user_id": 1, "fdc_id": 319878, "food_category_id": 5,  "meal_type": "lunch",
         "rating": 2.0, "would_eat_again": 0},
        {"user_id": 2, "fdc_id": 319900, "food_category_id": 11, "meal_type": "lunch",
         "rating": 5.0, "would_eat_again": 1},
        {"user_id": 2, "fdc_id": 319901, "food_category_id": 16, "meal_type": "breakfast",
         "rating": 4.0, "would_eat_again": 1},
        {"user_id": 3, "fdc_id": 319905, "food_category_id": 13, "meal_type": "dinner",
         "rating": 3.5, "would_eat_again": 1},
    ]

    logger.info("Submitting demo feedback events...")
    for event in demo_events:
        loop.submit_feedback(event)

    # -- Also seed from historical synthetic feedback -------------------
    hist_fb = pd.read_csv(PROCESSED_DIR / "feedback.csv")
    # Log a sample of 200 historical records
    sample = hist_fb.sample(200, random_state=42)
    for _, row in sample.iterrows():
        loop.store.log(row.to_dict())

    # -- Run full learning cycle ---------------------------------------
    loop.run_learning_cycle(min_new_records=20)

    # -- Show preference summary ---------------------------------------
    for uid in [1, 2, 3]:
        prefs    = loop.get_user_preferences(uid)
        affinities = loop.get_affinity_scores(uid)
        top_cats   = sorted(affinities, key=affinities.get, reverse=True)[:5]
        logger.info(f"\n  User {uid} preference summary:")
        logger.info(f"    Total feedback:     {prefs.get('total_feedback', 0)}")
        logger.info(f"    Avg rating:         {prefs.get('avg_rating', 0):.2f}")
        logger.info(f"    Liked foods:        {len(prefs.get('liked_fdc_ids', []))}")
        logger.info(f"    Disliked foods:     {len(prefs.get('disliked_fdc_ids', []))}")
        logger.info(f"    Top affinity cats:  {top_cats}")

    logger.info("\n[OK] PHASE 6 COMPLETE")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_feedback_demo()
