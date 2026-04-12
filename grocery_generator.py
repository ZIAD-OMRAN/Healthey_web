"""
PHASE 5: Grocery List Generator
Smart AI Diet & Meal Optimization System
=========================================
Reads a generated meal plan and produces:
  - Consolidated grocery list (by category, with quantities)
  - Cost estimates (heuristic unit-price database)
  - Printable text + structured JSON outputs
  - Deduplication across the week with sum of gram weights
"""

import json
import logging
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")

# -- Heuristic unit prices (USD per 100g) ---------------------------------
CATEGORY_PRICE_PER_100G = {
    "Dairy and Egg Products":               0.35,
    "Spices and Herbs":                     1.20,
    "Baby Foods":                           0.80,
    "Fats and Oils":                        0.40,
    "Poultry Products":                     0.55,
    "Soups, Sauces, and Gravies":           0.30,
    "Sausages and Luncheon Meats":          0.65,
    "Breakfast Cereals":                    0.45,
    "Fruits and Fruit Juices":              0.25,
    "Pork Products":                        0.60,
    "Vegetables and Vegetable Products":    0.20,
    "Nut and Seed Products":                0.90,
    "Beef Products":                        0.75,
    "Beverages":                            0.15,
    "Finfish and Shellfish Products":       0.85,
    "Legumes and Legume Products":          0.22,
    "Lamb, Veal, and Game Products":        0.90,
    "Baked Products":                       0.35,
    "Sweets":                               0.50,
    "Cereal Grains and Pasta":              0.20,
    "Fast Foods":                           0.60,
    "Meals, Entrees, and Side Dishes":      0.55,
    "Snacks":                               0.55,
    "American Indian/Alaska Native Foods":  0.40,
    "Restaurant Foods":                     0.65,
    "Branded Food Products Database":       0.50,
    "Quality Control Materials":            0.30,
    "Alcoholic Beverages":                  0.45,
    "Unknown":                              0.40,
}

UNITS_BY_CATEGORY = {
    "Dairy and Egg Products":    "units",
    "Beverages":                 "bottles",
    "Fruits and Fruit Juices":   "pieces",
    "Vegetables and Vegetable Products": "pieces",
    "Fats and Oils":             "bottle",
    "Spices and Herbs":          "jar",
}


# ===========================================================================
# CORE AGGREGATOR
# ===========================================================================

def aggregate_grocery_needs(plan: list[dict]) -> pd.DataFrame:
    """
    Flatten all foods in all days/meals -> group by (fdc_id, name, category)
    summing total gram weight needed for the week.
    """
    records = []
    for day in plan:
        for meal in day.get("meals", []):
            for food in meal.get("foods", []):
                records.append({
                    "fdc_id":      food["fdc_id"],
                    "name":        food["name"],
                    "category":    food.get("category", "Unknown"),
                    "serving_g":   food.get("serving_g", 100),
                    "energy_kcal": food.get("energy_kcal", 0),
                    "protein_g":   food.get("protein_g", 0),
                    "fat_g":       food.get("fat_g", 0),
                    "carbs_g":     food.get("carbs_g", 0),
                })
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    grocery = (
        df.groupby(["fdc_id", "name", "category"])
        .agg(
            occurrences=("serving_g", "count"),
            total_grams=("serving_g", "sum"),
            avg_kcal_per_serving=("energy_kcal", "mean"),
        )
        .reset_index()
    )
    # Round to sensible purchase quantities
    grocery["purchase_grams"] = (grocery["total_grams"] / 50).apply(
        lambda x: round(x) * 50
    ).clip(lower=50)
    return grocery


# ===========================================================================
# COST ESTIMATOR
# ===========================================================================

def estimate_costs(grocery: pd.DataFrame) -> pd.DataFrame:
    """Attach heuristic unit price and estimated weekly cost."""
    grocery = grocery.copy()
    grocery["price_per_100g"] = grocery["category"].map(CATEGORY_PRICE_PER_100G).fillna(0.40)
    grocery["estimated_cost_usd"] = (
        grocery["purchase_grams"] / 100.0 * grocery["price_per_100g"]
    ).round(2)
    grocery["unit"] = grocery["category"].map(UNITS_BY_CATEGORY).fillna("g")
    return grocery


# ===========================================================================
# FORMATTER
# ===========================================================================

def format_grocery_list(grocery: pd.DataFrame, user_name: str = "User") -> str:
    """Produce a printable, categorised grocery list."""
    lines = [
        "=" * 65,
        f"     SMART DIET SYSTEM - WEEKLY GROCERY LIST for {user_name}",
        "=" * 65,
    ]
    total_cost = 0.0
    for cat, grp in grocery.sort_values("category").groupby("category"):
        lines.append(f"\n    {cat}")
        lines.append("  " + "-" * 60)
        for _, row in grp.iterrows():
            qty_str = f"{int(row['purchase_grams'])}g"
            cost_str = f"${row['estimated_cost_usd']:.2f}"
            lines.append(
                f"    ?  {row['name'][:45]:<45} {qty_str:>6}  {cost_str:>7}"
            )
            total_cost += row["estimated_cost_usd"]
    lines.append("\n" + "-" * 65)
    lines.append(f"    ESTIMATED WEEKLY COST:  ${total_cost:.2f} USD")
    lines.append(
        f"  ?  TOTAL ITEMS:  {len(grocery)}  |  "
        f"TOTAL WEIGHT:  {grocery['purchase_grams'].sum() / 1000:.1f} kg"
    )
    lines.append("=" * 65)
    return "\n".join(lines)


def grocery_to_json(grocery: pd.DataFrame) -> dict:
    """Convert grocery DataFrame to structured JSON."""
    by_category = {}
    for cat, grp in grocery.groupby("category"):
        by_category[cat] = grp[[
            "fdc_id", "name", "purchase_grams", "occurrences",
            "estimated_cost_usd", "unit",
        ]].to_dict(orient="records")
    return {
        "total_items":        len(grocery),
        "total_weight_g":     int(grocery["purchase_grams"].sum()),
        "estimated_cost_usd": round(grocery["estimated_cost_usd"].sum(), 2),
        "by_category":        by_category,
    }


# ===========================================================================
# MAIN CLASS
# ===========================================================================

class GroceryListGenerator:
    def generate(self, plan_json_path: Path, user_name: str = "User") -> dict:
        """
        Read a saved meal plan JSON and generate grocery list.
        Returns dict with text report + structured data.
        """
        with open(plan_json_path) as f:
            plan_data = json.load(f)

        plan      = plan_data.get("plan", [])
        user_name = plan_data.get("user", {}).get("name", user_name)

        grocery = aggregate_grocery_needs(plan)
        if grocery.empty:
            logger.warning("No grocery data found in plan.")
            return {}

        grocery = estimate_costs(grocery)
        text    = format_grocery_list(grocery, user_name)
        data    = grocery_to_json(grocery)

        uid = plan_data.get("user", {}).get("user_id", "demo")
        txt_path  = PROCESSED_DIR / f"grocery_user_{uid}.txt"
        json_path = PROCESSED_DIR / f"grocery_user_{uid}.json"

        with open(txt_path, "w") as f:
            f.write(text)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Grocery list saved -> {txt_path}")
        logger.info(f"  Items: {data['total_items']} | "
                    f"Weight: {data['total_weight_g']/1000:.1f}kg | "
                    f"Cost: ${data['estimated_cost_usd']:.2f}")
        return {"text": text, "data": data, "grocery_df": grocery}


# ===========================================================================
# MAIN
# ===========================================================================

def run_grocery_demo():
    logger.info("=" * 60)
    logger.info("PHASE 5 - Grocery List Generator")
    logger.info("=" * 60)

    generator = GroceryListGenerator()
    for uid in [1, 2, 3]:
        plan_path = PROCESSED_DIR / f"meal_plan_user_{uid}.json"
        if plan_path.exists():
            result = generator.generate(plan_path)
            if result:
                logger.info(result["text"][:600])
                logger.info("  ...(truncated)\n")
        else:
            logger.warning(f"Plan not found for user {uid} - run Phase 4 first.")

    logger.info("[OK] PHASE 5 COMPLETE")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_grocery_demo()
