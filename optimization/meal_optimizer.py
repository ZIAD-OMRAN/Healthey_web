"""
PHASE 3: Optimization Engine - Nutrient-Constrained Meal Planning
Smart AI Diet & Meal Optimization System
==========================================
Uses PuLP (Linear Programming) + greedy heuristics to select foods that
optimally satisfy a user's daily nutritional targets while respecting:
  - Calorie budget
  - Macro targets (protein / fat / carbs)
  - Dietary restrictions (vegan, keto, etc.)
  - Allergen exclusions
  - Meal type distribution across the day
  - Variety constraints (no repeated foods)
"""

import logging
import numpy as np
import pandas as pd
import pulp
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

# -- Diet type -> allowed category ids ----------------------------------------
DIET_ALLOWED_CATEGORIES = {
    "omnivore":      None,   # All allowed
    "vegetarian":    [1, 2, 3, 4, 8, 9, 11, 12, 14, 16, 18, 19, 20, 23],
    "vegan":         [2, 4, 8, 9, 11, 12, 14, 16, 18, 19, 20, 23],
    "keto":          [1, 4, 5, 7, 10, 12, 13, 15, 17],   # high fat/protein, low carb
    "paleo":         [1, 4, 5, 9, 10, 11, 12, 13, 15, 17],
    "mediterranean": [1, 2, 4, 9, 11, 12, 14, 15, 16, 18, 20],
}

ALLERGEN_EXCLUDE_CATEGORIES = {
    "gluten":    [8, 18, 20],
    "dairy":     [1],
    "nut":       [12],
    "shellfish": [15],
    "none":      [],
}

MEAL_CALORIE_SPLIT = {
    "breakfast": 0.25,
    "lunch":     0.35,
    "dinner":    0.30,
    "snack":     0.10,
}


# ===========================================================================
# FOOD FILTER
# ===========================================================================

def filter_food_candidates(
    nutrition: pd.DataFrame,
    diet_type: str = "omnivore",
    food_allergy: str = "none",
    meal_type: str = "lunch",
    target_kcal: float = 600,
    max_candidates: int = 200,
) -> pd.DataFrame:
    """Return a filtered, scored set of candidate foods for a meal slot."""
    df = nutrition.copy()

    # -- Diet filter ------------------------------------------------------
    allowed_cats = DIET_ALLOWED_CATEGORIES.get(diet_type)
    if allowed_cats:
        df = df[df["food_category_id"].isin(allowed_cats)]

    # -- Allergen filter --------------------------------------------------
    excl_cats = ALLERGEN_EXCLUDE_CATEGORIES.get(food_allergy, [])
    if excl_cats:
        df = df[~df["food_category_id"].isin(excl_cats)]

    # -- Calorie range filter (+/-60% of slot target) -----------------------
    lo = target_kcal * 0.10
    hi = target_kcal * 1.60
    df = df[(df["energy_kcal"] >= lo) & (df["energy_kcal"] <= hi)]

    # -- Breakfast preference: lower calorie, higher carbs ----------------
    if meal_type == "breakfast":
        df = df[df["energy_kcal"] <= target_kcal * 1.2]

    # -- Score by nutrient density -----------------------------------------
    if "nutrient_density_score" in df.columns:
        df = df.nlargest(max_candidates, "nutrient_density_score")
    else:
        df = df.head(max_candidates)

    return df.reset_index(drop=True)


# ===========================================================================
# LP OPTIMIZER - single meal slot
# ===========================================================================

def optimize_meal_lp(
    candidates: pd.DataFrame,
    target_kcal: float,
    target_protein: float,
    target_fat: float,
    target_carbs: float,
    n_foods: int = 3,
    meal_label: str = "meal",
) -> Optional[pd.DataFrame]:
    """
    Solve a 0-1 ILP to select exactly n_foods from candidates that
    minimise deviation from macro targets.

    Returns selected foods dataframe or None if infeasible.
    """
    if len(candidates) < n_foods:
        logger.warning(f"  [{meal_label}] Only {len(candidates)} candidates (need {n_foods})")
        n_foods = max(1, len(candidates))

    # -- Build problem ----------------------------------------------------
    prob = pulp.LpProblem(f"meal_{meal_label}", pulp.LpMinimize)
    n    = len(candidates)
    x    = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # -- Extract nutrient vectors ------------------------------------------
    kcal    = candidates["energy_kcal"].fillna(0).values
    protein = candidates["protein_g"].fillna(0).values
    fat     = candidates["fat_g"].fillna(0).values
    carbs   = candidates["carbs_g"].fillna(0).values

    # -- Slack variables for soft constraints -----------------------------
    slack_kcal_over  = pulp.LpVariable("s_kcal_over",  lowBound=0)
    slack_kcal_under = pulp.LpVariable("s_kcal_under", lowBound=0)
    slack_prot_under = pulp.LpVariable("s_prot_under", lowBound=0)
    slack_fat_over   = pulp.LpVariable("s_fat_over",   lowBound=0)
    slack_carb_over  = pulp.LpVariable("s_carb_over",  lowBound=0)

    # -- Objective: minimise weighted deviations --------------------------
    prob += (
        2.0 * slack_kcal_over
        + 2.0 * slack_kcal_under
        + 1.5 * slack_prot_under
        + 1.0 * slack_fat_over
        + 1.0 * slack_carb_over
    )

    # -- Constraints ------------------------------------------------------
    total_kcal    = pulp.lpSum(kcal[i]    * x[i] for i in range(n))
    total_protein = pulp.lpSum(protein[i] * x[i] for i in range(n))
    total_fat     = pulp.lpSum(fat[i]     * x[i] for i in range(n))
    total_carbs   = pulp.lpSum(carbs[i]   * x[i] for i in range(n))

    # Calorie soft bounds
    prob += total_kcal - slack_kcal_over  <= target_kcal * 1.10
    prob += total_kcal + slack_kcal_under >= target_kcal * 0.90

    # Protein minimum (soft)
    prob += total_protein + slack_prot_under >= target_protein * 0.85

    # Fat ceiling (soft)
    prob += total_fat - slack_fat_over <= target_fat * 1.20

    # Carb ceiling (soft)
    prob += total_carbs - slack_carb_over <= target_carbs * 1.20

    # Exactly n_foods selected
    prob += pulp.lpSum(x) == n_foods

    # -- Solve silently ----------------------------------------------------
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=10)
    status = prob.solve(solver)

    if pulp.LpStatus[status] not in ("Optimal", "Not Solved"):
        logger.warning(f"  [{meal_label}] LP infeasible ({pulp.LpStatus[status]}); using greedy fallback")
        return _greedy_fallback(candidates, target_kcal, target_protein, n_foods)

    selected_idx = [i for i in range(n) if pulp.value(x[i]) == 1.0]
    if not selected_idx:
        return _greedy_fallback(candidates, target_kcal, target_protein, n_foods)

    result = candidates.iloc[selected_idx].copy()
    result["meal_type"] = meal_label
    return result


def _greedy_fallback(
    candidates: pd.DataFrame,
    target_kcal: float,
    target_protein: float,
    n_foods: int,
) -> pd.DataFrame:
    """Simple greedy selection: pick best nutrient-density items within kcal budget."""
    df = candidates.copy().sort_values("nutrient_density_score", ascending=False)
    selected, total_kcal = [], 0.0
    for _, row in df.iterrows():
        if len(selected) >= n_foods:
            break
        if total_kcal + row["energy_kcal"] <= target_kcal * 1.15:
            selected.append(row)
            total_kcal += row["energy_kcal"]
    if not selected:
        selected = [df.iloc[0]]
    result = pd.DataFrame(selected)
    result["meal_type"] = "greedy_fallback"
    return result


# ===========================================================================
# DAILY MEAL OPTIMIZER
# ===========================================================================

def optimize_daily_plan(
    nutrition: pd.DataFrame,
    user_targets: dict,
    diet_type: str = "omnivore",
    food_allergy: str = "none",
    meals_per_day: int = 3,
    snacks_per_day: int = 1,
) -> dict:
    """
    Orchestrates LP optimisation across all meal slots for one day.
    Returns a dict of {meal_type: selected_foods_df}.
    """
    target_kcal    = user_targets["target_kcal"]
    target_protein = user_targets["target_protein_g"]
    target_fat     = user_targets["target_fat_g"]
    target_carbs   = user_targets["target_carbs_g"]

    # Determine active meal slots
    slots = []
    if meals_per_day >= 1: slots.append("breakfast")
    if meals_per_day >= 2: slots.append("lunch")
    if meals_per_day >= 3: slots.append("dinner")
    for s in range(snacks_per_day):
        slots.append(f"snack_{s+1}")

    # Re-distribute calorie split across active slots
    base_splits = {
        "breakfast": 0.25, "lunch": 0.35, "dinner": 0.30,
        "snack_1": 0.05, "snack_2": 0.05,
    }
    total_split = sum(base_splits.get(s, 0.05) for s in slots)
    normalised  = {s: base_splits.get(s, 0.05) / total_split for s in slots}

    daily_plan = {}
    used_fdc_ids = set()

    for slot in slots:
        frac         = normalised[slot]
        slot_kcal    = target_kcal    * frac
        slot_protein = target_protein * frac
        slot_fat     = target_fat     * frac
        slot_carbs   = target_carbs   * frac
        n_foods      = 2 if "snack" in slot else 3

        candidates = filter_food_candidates(
            nutrition, diet_type, food_allergy, slot, slot_kcal, max_candidates=150
        )
        # Exclude already-used foods for variety
        candidates = candidates[~candidates["fdc_id"].isin(used_fdc_ids)]
        if len(candidates) == 0:
            logger.warning(f"  No candidates left for slot {slot}")
            continue

        selected = optimize_meal_lp(
            candidates, slot_kcal, slot_protein, slot_fat, slot_carbs,
            n_foods=n_foods, meal_label=slot,
        )
        if selected is not None and len(selected) > 0:
            selected["meal_slot"] = slot
            daily_plan[slot]      = selected
            used_fdc_ids.update(selected["fdc_id"].tolist())

    # -- Daily nutrition totals --------------------------------------------
    if daily_plan:
        all_foods = pd.concat(daily_plan.values(), ignore_index=True)
        totals = {
            "total_kcal":    all_foods["energy_kcal"].sum(),
            "total_protein": all_foods["protein_g"].sum(),
            "total_fat":     all_foods["fat_g"].sum(),
            "total_carbs":   all_foods["carbs_g"].sum(),
            "target_kcal":   target_kcal,
            "target_protein": target_protein,
            "target_fat":    target_fat,
            "target_carbs":  target_carbs,
        }
        logger.info(f"  Daily totals -> kcal:{totals['total_kcal']:.0f}/{target_kcal:.0f} | "
                    f"prot:{totals['total_protein']:.1f}/{target_protein:.0f}g | "
                    f"fat:{totals['total_fat']:.1f}/{target_fat:.0f}g | "
                    f"carbs:{totals['total_carbs']:.1f}/{target_carbs:.0f}g")
        daily_plan["__totals__"] = totals

    return daily_plan


# ===========================================================================
# WEEKLY PLAN OPTIMIZER
# ===========================================================================

def optimize_weekly_plan(
    nutrition: pd.DataFrame,
    user_targets: dict,
    diet_type: str = "omnivore",
    food_allergy: str = "none",
    meals_per_day: int = 3,
    snacks_per_day: int = 1,
    days: int = 7,
) -> list[dict]:
    """Generate a 7-day optimised meal plan."""
    weekly = []
    DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for d in range(days):
        logger.info(f"  Optimising {DAY_NAMES[d]}...")
        plan = optimize_daily_plan(
            nutrition, user_targets, diet_type, food_allergy, meals_per_day, snacks_per_day
        )
        plan["__day__"] = DAY_NAMES[d]
        weekly.append(plan)
    return weekly


# ===========================================================================
# MAIN
# ===========================================================================

def run_optimization_demo():
    logger.info("=" * 60)
    logger.info("PHASE 3 - Optimization Engine")
    logger.info("=" * 60)

    nutrition = pd.read_csv(PROCESSED_DIR / "nutrition_full.csv", low_memory=False)

    # Demo user targets
    demo_targets = {
        "target_kcal":      2000,
        "target_protein_g": 150,
        "target_fat_g":     65,
        "target_carbs_g":   220,
    }

    logger.info("Running 7-day optimisation for demo user (omnivore)...")
    weekly = optimize_weekly_plan(
        nutrition, demo_targets,
        diet_type="omnivore", food_allergy="none",
        meals_per_day=3, snacks_per_day=1,
    )

    # Quick summary
    day_totals = []
    for day in weekly:
        t = day.get("__totals__", {})
        if t:
            day_totals.append({
                "day":     day.get("__day__", "?"),
                "kcal":    round(t["total_kcal"], 1),
                "protein": round(t["total_protein"], 1),
                "fat":     round(t["total_fat"], 1),
                "carbs":   round(t["total_carbs"], 1),
            })
    summary_df = pd.DataFrame(day_totals)
    logger.info(f"\n7-Day Plan Summary:\n{summary_df.to_string(index=False)}")
    summary_df.to_csv(PROCESSED_DIR / "weekly_plan_summary.csv", index=False)
    logger.info("\n[OK] PHASE 3 COMPLETE")
    return weekly


if __name__ == "__main__":
    run_optimization_demo()
