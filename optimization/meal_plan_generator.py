"""
PHASE 4: Meal Plan Generator
Smart AI Diet & Meal Optimization System
=========================================
Integrates Phase 2 (ML predictions) + Phase 3 (LP optimiser) to produce:
  - Personalised weekly meal plans
  - Per-meal nutrient breakdowns
  - Human-readable plan reports
  - Structured JSON for API consumption
"""

import logging
import json
import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

from optimization.meal_optimizer import optimize_weekly_plan

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
OUTPUT_DIR    = Path("data/processed")


# ===========================================================================
# USER TARGET PREDICTOR (wraps Phase 2 model)
# ===========================================================================

class UserTargetPredictor:
    """
    Loads the trained UserNutritionNet and predicts personalised macro targets.
    Falls back to Harris-Benedict equation if model unavailable.
    """

    def __init__(self):
        try:
            from models.neural_networks import UserNutritionNet, DEVICE
            self.feat_scaler   = joblib.load(MODELS_DIR / "user_feat_scaler.pkl")
            self.target_scaler = joblib.load(MODELS_DIR / "user_target_scaler.pkl")
            self.encoders      = joblib.load(MODELS_DIR / "user_encoders.pkl")
            feature_cols = self.encoders["feature_cols"]
            self.model = UserNutritionNet(input_dim=len(feature_cols)).to(DEVICE)
            self.model.load_state_dict(
                torch.load(MODELS_DIR / "user_nutrition_net.pt",
                           map_location=DEVICE, weights_only=True)
            )
            self.model.eval()
            self.DEVICE = DEVICE
            self._use_model = True
            logger.info("UserNutritionNet loaded for target prediction.")
        except Exception as e:
            logger.warning(f"Could not load NN model ({e}); falling back to Harris-Benedict.")
            self._use_model = False

    def predict(self, user: dict) -> dict:
        if self._use_model:
            return self._nn_predict(user)
        return self._hb_predict(user)

    def _nn_predict(self, user: dict) -> dict:
        enc = self.encoders
        try:
            feats = np.array([[
                user.get("age", 30),
                user.get("weight_kg", 70),
                user.get("height_cm", 170),
                user.get("bmr", 1600),
                user.get("tdee", 2000),
                user.get("meals_per_day", 3),
                user.get("snacks_per_day", 1),
                enc["goal"].transform([user.get("dietary_goal", "maintenance")])[0],
                enc["activity"].transform([user.get("activity_level", "moderate")])[0],
                enc["diet"].transform([user.get("diet_type", "omnivore")])[0],
                enc["gender"].transform([user.get("gender", "male")])[0],
                enc["allergy"].transform([user.get("food_allergy", "none")])[0],
            ]], dtype=np.float32)
            feats_sc = self.feat_scaler.transform(feats)
            with torch.no_grad():
                pred = self.model(
                    torch.tensor(feats_sc).to(self.DEVICE)
                ).cpu().numpy()
            raw = self.target_scaler.inverse_transform(pred)[0]
            return {
                "target_kcal":      max(float(raw[0]), 800),
                "target_protein_g": max(float(raw[1]), 30),
                "target_fat_g":     max(float(raw[2]), 20),
                "target_carbs_g":   max(float(raw[3]), 20),
            }
        except Exception as e:
            logger.warning(f"NN predict failed ({e}); using HB fallback.")
            return self._hb_predict(user)

    @staticmethod
    def _hb_predict(user: dict) -> dict:
        """Harris-Benedict BMR + activity multiplier + goal adjustment."""
        w, h, a = user.get("weight_kg", 70), user.get("height_cm", 170), user.get("age", 30)
        gender  = user.get("gender", "male")
        if gender == "male":
            bmr = 10 * w + 6.25 * h - 5 * a + 5
        else:
            bmr = 10 * w + 6.25 * h - 5 * a - 161
        tdee_mult = {
            "sedentary": 1.2, "light": 1.375, "moderate": 1.55,
            "active": 1.725, "very_active": 1.9,
        }.get(user.get("activity_level", "moderate"), 1.55)
        tdee = bmr * tdee_mult
        goal_mult = {
            "weight_loss": 0.80, "muscle_gain": 1.10,
            "maintenance": 1.00, "heart_health": 0.95, "diabetic_control": 0.90,
        }.get(user.get("dietary_goal", "maintenance"), 1.00)
        target_kcal    = round(tdee * goal_mult)
        target_protein = round(w * 1.6)
        target_fat     = round(target_kcal * 0.25 / 9, 1)
        target_carbs   = max(round((target_kcal - target_protein * 4 - target_fat * 9) / 4, 1), 20)
        return {
            "target_kcal":      target_kcal,
            "target_protein_g": target_protein,
            "target_fat_g":     target_fat,
            "target_carbs_g":   target_carbs,
        }


# ===========================================================================
# MEAL PLAN FORMATTER
# ===========================================================================

def format_meal(slot: str, foods_df: pd.DataFrame) -> dict:
    """Convert a selected-foods DataFrame into a structured meal dict."""
    foods_list = []
    for _, row in foods_df.iterrows():
        foods_list.append({
            "fdc_id":      int(row.get("fdc_id", 0)),
            "name":        str(row.get("description", "Unknown Food"))[:60],
            "category":    str(row.get("category_name", "?")),
            "serving_g":   round(float(row.get("serving_gram_weight", 100)), 1),
            "energy_kcal": round(float(row.get("energy_kcal", 0)), 1),
            "protein_g":   round(float(row.get("protein_g", 0)), 1),
            "fat_g":       round(float(row.get("fat_g", 0)), 1),
            "carbs_g":     round(float(row.get("carbs_g", 0)), 1),
            "fiber_g":     round(float(row.get("fiber_g", 0)), 1),
        })
    totals = {
        "energy_kcal": round(foods_df["energy_kcal"].sum(), 1),
        "protein_g":   round(foods_df["protein_g"].sum(), 1),
        "fat_g":       round(foods_df["fat_g"].sum(), 1),
        "carbs_g":     round(foods_df["carbs_g"].sum(), 1),
    }
    return {"slot": slot, "foods": foods_list, "totals": totals}


def format_daily_plan(day_name: str, day_dict: dict, day_date: str) -> dict:
    """Format a full day's plan into a structured dict."""
    meals = []
    for slot, val in day_dict.items():
        if slot.startswith("__"):
            continue
        if isinstance(val, pd.DataFrame) and len(val) > 0:
            meals.append(format_meal(slot, val))

    totals = day_dict.get("__totals__", {})
    return {
        "day":    day_name,
        "date":   day_date,
        "meals":  meals,
        "totals": {
            "energy_kcal": round(totals.get("total_kcal", 0), 1),
            "protein_g":   round(totals.get("total_protein", 0), 1),
            "fat_g":       round(totals.get("total_fat", 0), 1),
            "carbs_g":     round(totals.get("total_carbs", 0), 1),
        },
        "targets": {
            "energy_kcal": round(totals.get("target_kcal", 0), 1),
            "protein_g":   round(totals.get("target_protein", 0), 1),
            "fat_g":       round(totals.get("target_fat", 0), 1),
            "carbs_g":     round(totals.get("target_carbs", 0), 1),
        },
    }


# ===========================================================================
# PLAN REPORT GENERATOR
# ===========================================================================

def generate_text_report(weekly_plan: list[dict], user: dict, targets: dict) -> str:
    """Produce a human-readable weekly meal plan report."""
    lines = [
        "=" * 70,
        "       SMART AI DIET SYSTEM - PERSONALISED WEEKLY MEAL PLAN",
        "=" * 70,
        f"  User:     {user.get('name', 'User')}",
        f"  Goal:     {user.get('dietary_goal', '?').replace('_', ' ').title()}",
        f"  Diet:     {user.get('diet_type', '?').title()}",
        f"  Targets:  {targets['target_kcal']:.0f} kcal | "
        f"{targets['target_protein_g']:.0f}g protein | "
        f"{targets['target_fat_g']:.0f}g fat | "
        f"{targets['target_carbs_g']:.0f}g carbs",
        "=" * 70,
    ]

    for day_plan in weekly_plan:
        lines.append(f"\n{'-'*70}")
        lines.append(f"    {day_plan['day'].upper()}  ({day_plan['date']})")
        lines.append(f"{'-'*70}")
        for meal in day_plan["meals"]:
            slot_label = meal["slot"].replace("_", " ").title()
            lines.append(f"\n    {slot_label}  "
                         f"({meal['totals']['energy_kcal']:.0f} kcal | "
                         f"{meal['totals']['protein_g']:.0f}g P | "
                         f"{meal['totals']['fat_g']:.0f}g F | "
                         f"{meal['totals']['carbs_g']:.0f}g C)")
            for food in meal["foods"]:
                lines.append(f"       - {food['name'][:55]:<55} "
                              f"{food['energy_kcal']:>5.0f} kcal "
                              f"({food['serving_g']:.0f}g)")

        t = day_plan["totals"]
        g = day_plan["targets"]
        lines.append(f"\n   Daily total: "
                     f"{t['energy_kcal']:.0f}/{g['energy_kcal']:.0f} kcal | "
                     f"P {t['protein_g']:.0f}/{g['protein_g']:.0f}g | "
                     f"F {t['fat_g']:.0f}/{g['fat_g']:.0f}g | "
                     f"C {t['carbs_g']:.0f}/{g['carbs_g']:.0f}g")

    lines.append("\n" + "=" * 70)
    lines.append("  Generated by Smart AI Diet & Meal Optimization System")
    lines.append("=" * 70)
    return "\n".join(lines)


# ===========================================================================
# MAIN GENERATOR
# ===========================================================================

class MealPlanGenerator:
    def __init__(self):
        self.nutrition   = pd.read_csv(PROCESSED_DIR / "nutrition_full.csv", low_memory=False)
        self.predictor   = UserTargetPredictor()
        logger.info(f"MealPlanGenerator ready | {len(self.nutrition):,} foods loaded")

    def generate(
        self,
        user: dict,
        days: int = 7,
        start_date: Optional[date] = None,
    ) -> dict:
        """
        Full pipeline: user profile -> targets -> weekly optimised plan.
        Returns a structured dict with plan data + text report.
        """
        if start_date is None:
            start_date = date.today()

        # -- Step 1: Predict targets --------------------------------------
        targets = self.predictor.predict(user)
        logger.info(f"Targets: kcal={targets['target_kcal']:.0f} | "
                    f"P={targets['target_protein_g']:.0f}g | "
                    f"F={targets['target_fat_g']:.0f}g | "
                    f"C={targets['target_carbs_g']:.0f}g")

        # -- Step 2: Optimise weekly plan ---------------------------------
        weekly_raw = optimize_weekly_plan(
            self.nutrition,
            targets,
            diet_type     = user.get("diet_type", "omnivore"),
            food_allergy  = user.get("food_allergy", "none"),
            meals_per_day = user.get("meals_per_day", 3),
            snacks_per_day= user.get("snacks_per_day", 1),
            days=days,
        )

        # -- Step 3: Format output ----------------------------------------
        DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        weekly_formatted = []
        for i, day_dict in enumerate(weekly_raw):
            day_name = DAY_NAMES[i % 7]
            day_date = (start_date + timedelta(days=i)).isoformat()
            weekly_formatted.append(format_daily_plan(day_name, day_dict, day_date))

        # -- Step 4: Generate report --------------------------------------
        report = generate_text_report(weekly_formatted, user, targets)

        result = {
            "user":    user,
            "targets": targets,
            "plan":    weekly_formatted,
            "report":  report,
        }

        # -- Step 5: Save outputs ------------------------------------------
        uid = user.get("user_id", "demo")
        plan_path   = OUTPUT_DIR / f"meal_plan_user_{uid}.json"
        report_path = OUTPUT_DIR / f"meal_plan_user_{uid}.txt"

        with open(plan_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Plan saved -> {plan_path}")
        logger.info(f"Report saved -> {report_path}")
        return result


# ===========================================================================
# MAIN
# ===========================================================================

def run_meal_plan_demo():
    logger.info("=" * 60)
    logger.info("PHASE 4 - Meal Plan Generator")
    logger.info("=" * 60)

    generator = MealPlanGenerator()

    demo_users = [
        {
            "user_id": 1, "name": "Ahmed (Weight Loss)",
            "age": 30, "weight_kg": 90, "height_cm": 178, "gender": "male",
            "activity_level": "moderate", "dietary_goal": "weight_loss",
            "diet_type": "omnivore", "food_allergy": "none",
            "meals_per_day": 3, "snacks_per_day": 1,
        },
        {
            "user_id": 2, "name": "Sara (Vegan Muscle)",
            "age": 26, "weight_kg": 62, "height_cm": 165, "gender": "female",
            "activity_level": "active", "dietary_goal": "muscle_gain",
            "diet_type": "vegan", "food_allergy": "none",
            "meals_per_day": 4, "snacks_per_day": 2,
        },
        {
            "user_id": 3, "name": "Kareem (Keto + Diabetes)",
            "age": 45, "weight_kg": 85, "height_cm": 175, "gender": "male",
            "activity_level": "light", "dietary_goal": "diabetic_control",
            "diet_type": "keto", "food_allergy": "gluten",
            "meals_per_day": 3, "snacks_per_day": 0,
        },
    ]

    for user in demo_users:
        logger.info(f"\nGenerating plan for: {user['name']}")
        result = generator.generate(user, days=7)
        logger.info(result["report"][:800])
        logger.info("  ...(truncated)")

    logger.info("\n[OK] PHASE 4 COMPLETE - meal plans generated")


if __name__ == "__main__":
    run_meal_plan_demo()
