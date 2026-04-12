"""
PHASE 1: Data Analysis & Preprocessing Pipeline
Smart AI Diet & Meal Optimization System
=========================================
Loads, merges, cleans, and engineers features from the USDA FoodData Central
dataset into a single unified nutrition matrix ready for ML training.
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/preprocessing.log"),
    ],
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -- Key nutrients we care about ---------------------------------------------
NUTRIENT_MAP = {
    1008: "energy_kcal",
    1003: "protein_g",
    1004: "fat_g",
    1005: "carbs_g",
    1079: "fiber_g",
    2000: "sugars_g",
    1093: "sodium_mg",
    1087: "calcium_mg",
    1089: "iron_mg",
    1114: "vitamin_d_ug",
    1162: "vitamin_c_mg",
    1175: "vitamin_b6_mg",
    1178: "vitamin_b12_ug",
    1109: "vitamin_e_mg",
    1106: "vitamin_a_ug",
    1092: "potassium_mg",
    1091: "phosphorus_mg",
    1090: "magnesium_mg",
    1095: "zinc_mg",
    1253: "cholesterol_mg",
    1258: "saturated_fat_g",
    1292: "monounsaturated_fat_g",
    1293: "polyunsaturated_fat_g",
}


# ===========================================================================
# STEP 1 - Load raw tables
# ===========================================================================

def load_raw_tables() -> dict[str, pd.DataFrame]:
    """Load all relevant CSV tables from the raw data directory."""
    files = {
        "food": "food.csv",
        "nutrient": "nutrient.csv",
        "food_nutrient": "food_nutrient.csv",
        "food_category": "food_category.csv",
        "food_portion": "food_portion.csv",
        "food_component": "food_component.csv",
        "measure_unit": "measure_unit.csv",
        "food_attribute": "food_attribute.csv",
        "food_attribute_type": "food_attribute_type.csv",
        "food_calorie_conversion_factor": "food_calorie_conversion_factor.csv",
        "market_acquisition": "market_acquisition.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, fname in files.items():
        path = RAW_DIR / fname
        if path.exists():
            tables[key] = pd.read_csv(path, low_memory=False)
            logger.info(f"Loaded {fname}: {tables[key].shape}")
        else:
            logger.warning(f"Missing file: {fname}")
    return tables


# ===========================================================================
# STEP 2 - Build wide nutrition matrix (foods x nutrients)
# ===========================================================================

def build_nutrition_matrix(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Pivot food_nutrient into a wide matrix:
    rows = foods (fdc_id), columns = key nutrients.
    """
    fn = tables["food_nutrient"][["fdc_id", "nutrient_id", "amount"]].copy()
    fn = fn[fn["nutrient_id"].isin(NUTRIENT_MAP.keys())]
    fn["nutrient_id"] = fn["nutrient_id"].map(NUTRIENT_MAP)

    # Aggregate duplicates (some foods appear multiple times per nutrient)
    fn = fn.groupby(["fdc_id", "nutrient_id"], as_index=False)["amount"].mean()

    matrix = fn.pivot(index="fdc_id", columns="nutrient_id", values="amount").reset_index()
    matrix.columns.name = None
    logger.info(f"Nutrition matrix shape: {matrix.shape}")
    return matrix


# ===========================================================================
# STEP 3 - Merge food metadata
# ===========================================================================

def merge_food_metadata(matrix: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join nutrition matrix with food description and category."""
    food = tables["food"][["fdc_id", "description", "food_category_id", "data_type"]].copy()
    category = tables["food_category"][["id", "description"]].rename(
        columns={"id": "food_category_id", "description": "category_name"}
    )
    food = food.merge(category, on="food_category_id", how="left")
    merged = matrix.merge(food, on="fdc_id", how="left")
    logger.info(f"After metadata merge: {merged.shape}")
    return merged


# ===========================================================================
# STEP 4 - Add portion data (default gram weight)
# ===========================================================================

def add_portion_data(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Attach a representative serving size (gram weight) per food."""
    portion = tables["food_portion"][["fdc_id", "gram_weight", "portion_description", "amount"]].copy()
    portion = portion.dropna(subset=["gram_weight"])
    # Use the first listed portion per food as the default serving
    portion_default = (
        portion.sort_values("gram_weight")
        .groupby("fdc_id", as_index=False)
        .first()
        .rename(columns={
            "gram_weight": "serving_gram_weight",
            "portion_description": "serving_description",
            "amount": "serving_amount",
        })
    )
    df = df.merge(portion_default[["fdc_id", "serving_gram_weight", "serving_description"]], 
                  on="fdc_id", how="left")
    # Fill missing with 100 g (per-100 g basis is the dataset norm)
    df["serving_gram_weight"] = df["serving_gram_weight"].fillna(100.0)
    logger.info(f"After portion merge: {df.shape}")
    return df


# ===========================================================================
# STEP 5 - Clean & impute
# ===========================================================================

def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop rows with no description.
    - Drop rows missing ALL nutrient values.
    - Impute remaining missing nutrients with category medians, else 0.
    - Remove physiologically impossible values (negative amounts).
    """
    df = df.dropna(subset=["description"])
    nutrient_cols = list(NUTRIENT_MAP.values())
    existing_nutrient_cols = [c for c in nutrient_cols if c in df.columns]

    # Drop rows where every nutrient is NaN
    df = df.dropna(subset=existing_nutrient_cols, how="all")

    # Clip negatives to 0
    for col in existing_nutrient_cols:
        df[col] = df[col].clip(lower=0)

    # Cap extreme outliers (>99.9th percentile per nutrient)
    for col in existing_nutrient_cols:
        cap = df[col].quantile(0.999)
        df[col] = df[col].clip(upper=cap)

    # Impute missing with category median, else global median
    for col in existing_nutrient_cols:
        cat_median = df.groupby("category_name")[col].transform("median")
        global_median = df[col].median()
        df[col] = df[col].fillna(cat_median).fillna(global_median).fillna(0)

    logger.info(f"After cleaning: {df.shape}")
    logger.info(f"Null counts:\n{df[existing_nutrient_cols].isnull().sum()}")
    return df


# ===========================================================================
# STEP 6 - Feature engineering
# ===========================================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive composite nutrition features useful for ML and diet scoring:
      - nutrient_density_score  - vitamins + minerals normalised by calories
      - protein_per_calorie     - satiety proxy
      - carb_to_fiber_ratio     - glycaemic quality proxy
      - fat_quality_ratio       - unsaturated / total fat
      - macro_balance_score     - how balanced the macro distribution is
      - diet_tags               - binary flags per dietary profile
    """
    eps = 1e-6

    # -- Nutrient density (higher = more micronutrients per kcal) ------------
    micro_cols = [c for c in [
        "calcium_mg", "iron_mg", "vitamin_c_mg", "vitamin_d_ug",
        "vitamin_a_ug", "potassium_mg", "magnesium_mg", "zinc_mg"
    ] if c in df.columns]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    if micro_cols:
        micro_scaled = scaler.fit_transform(df[micro_cols].fillna(0))
        df["nutrient_density_score"] = micro_scaled.sum(axis=1) / (df["energy_kcal"].fillna(eps) + eps) * 100
    else:
        df["nutrient_density_score"] = 0.0

    # -- Protein per calorie -------------------------------------------------
    df["protein_per_calorie"] = df["protein_g"] / (df["energy_kcal"] + eps)

    # -- Carb-to-fiber ratio (lower is better for glycaemic control) ---------
    if "fiber_g" in df.columns:
        df["carb_to_fiber_ratio"] = df["carbs_g"] / (df["fiber_g"] + eps)
        df["carb_to_fiber_ratio"] = df["carb_to_fiber_ratio"].clip(0, 100)
    else:
        df["carb_to_fiber_ratio"] = df["carbs_g"]

    # -- Fat quality (unsaturated proportion) --------------------------------
    unsat_fat = 0.0
    if "monounsaturated_fat_g" in df.columns:
        unsat_fat += df["monounsaturated_fat_g"]
    if "polyunsaturated_fat_g" in df.columns:
        unsat_fat += df["polyunsaturated_fat_g"]
    df["fat_quality_ratio"] = unsat_fat / (df["fat_g"] + eps)
    df["fat_quality_ratio"] = df["fat_quality_ratio"].clip(0, 1)

    # -- Macro balance score (entropy-based; higher = more balanced) ---------
    def macro_entropy(row):
        total = row["protein_g"] + row["fat_g"] + row["carbs_g"] + eps
        p = np.array([row["protein_g"], row["fat_g"], row["carbs_g"]]) / total
        p = p[p > 0]
        return -np.sum(p * np.log(p + eps))
    df["macro_balance_score"] = df.apply(macro_entropy, axis=1)

    # -- Diet tags (binary) --------------------------------------------------
    df["is_high_protein"] = (df["protein_g"] >= 20).astype(int)
    df["is_low_carb"]     = (df["carbs_g"] <= 10).astype(int)
    df["is_low_fat"]      = (df["fat_g"] <= 3).astype(int)
    df["is_low_calorie"]  = (df["energy_kcal"] <= 100).astype(int)
    df["is_high_fiber"]   = (df["fiber_g"] >= 5).astype(int) if "fiber_g" in df.columns else 0
    df["is_low_sodium"]   = (df["sodium_mg"] <= 140).astype(int) if "sodium_mg" in df.columns else 0

    logger.info("Feature engineering complete.")
    return df


# ===========================================================================
# STEP 7 - Encode categories & normalise
# ===========================================================================

def encode_and_normalise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_encoded  - ML-ready numeric dataframe (no text columns)
      df_full     - full dataframe with description and category kept
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    df_full = df.copy()

    # Encode food category
    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category_name"].fillna("Unknown"))

    # Encode data_type
    df["data_type_encoded"] = le.fit_transform(df["data_type"].fillna("unknown"))

    # Select numeric feature columns for ML
    nutrient_cols = list(NUTRIENT_MAP.values())
    feature_cols = (
        [c for c in nutrient_cols if c in df.columns]
        + [
            "nutrient_density_score", "protein_per_calorie",
            "carb_to_fiber_ratio", "fat_quality_ratio",
            "macro_balance_score", "serving_gram_weight",
            "is_high_protein", "is_low_carb", "is_low_fat",
            "is_low_calorie", "is_high_fiber", "is_low_sodium",
            "category_encoded",
        ]
    )
    feature_cols = [c for c in feature_cols if c in df.columns]

    df_encoded = df[["fdc_id"] + feature_cols].copy()

    # Z-score normalise continuous columns (exclude binary flags and encoded categoricals)
    binary_cols = [c for c in feature_cols if c.startswith("is_") or c.endswith("_encoded")]
    scale_cols  = [c for c in feature_cols if c not in binary_cols]
    scaler = StandardScaler()
    df_encoded[scale_cols] = scaler.fit_transform(df_encoded[scale_cols].fillna(0))

    import joblib
    joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")
    logger.info(f"Scaler saved. Encoded shape: {df_encoded.shape}")
    return df_encoded, df_full


# ===========================================================================
# STEP 8 - Generate synthetic user profiles
# ===========================================================================

def generate_synthetic_users(n: int = 5000) -> pd.DataFrame:
    """
    Since no real user_behavior.csv was provided, we synthesise realistic
    user profiles covering diverse dietary goals, restrictions, and preferences.
    This data drives the neural network training in Phase 2.
    """
    rng = np.random.default_rng(42)

    goals       = ["weight_loss", "muscle_gain", "maintenance", "heart_health", "diabetic_control"]
    diet_types  = ["omnivore", "vegetarian", "vegan", "keto", "paleo", "mediterranean"]
    activity    = ["sedentary", "light", "moderate", "active", "very_active"]
    allergies   = ["none", "gluten", "dairy", "nut", "shellfish"]
    genders     = ["male", "female", "non_binary"]

    tdee_map = {
        "sedentary": 1.2, "light": 1.375, "moderate": 1.55,
        "active": 1.725, "very_active": 1.9,
    }
    bmr_base_male   = 1600  # rough avg kcal BMR
    bmr_base_female = 1400

    rows = []
    for uid in range(1, n + 1):
        gender = rng.choice(genders)
        age    = int(rng.integers(18, 75))
        weight = round(float(rng.uniform(50, 120)), 1)   # kg
        height = round(float(rng.uniform(155, 200)), 1)  # cm
        act    = rng.choice(activity)
        goal   = rng.choice(goals)
        diet   = rng.choice(diet_types)
        allergy = rng.choice(allergies)

        bmr_base = bmr_base_male if gender == "male" else bmr_base_female
        bmr  = round(10 * weight + 6.25 * height - 5 * age + (5 if gender == "male" else -161), 1)
        tdee = round(bmr * tdee_map[act], 1)

        # Calorie targets
        if goal == "weight_loss":
            target_kcal = round(tdee * 0.80)
            target_protein = round(weight * 1.6)
        elif goal == "muscle_gain":
            target_kcal = round(tdee * 1.10)
            target_protein = round(weight * 2.0)
        elif goal == "heart_health":
            target_kcal = round(tdee * 0.95)
            target_protein = round(weight * 1.2)
        elif goal == "diabetic_control":
            target_kcal = round(tdee * 0.90)
            target_protein = round(weight * 1.4)
        else:
            target_kcal = round(tdee)
            target_protein = round(weight * 1.2)

        target_fat   = round(target_kcal * 0.25 / 9, 1)
        target_carbs = round((target_kcal - target_protein * 4 - target_fat * 9) / 4, 1)

        # Satisfaction rating (feedback proxy) - driven by goal + activity match
        base_satisfaction = rng.uniform(2.5, 5.0)

        rows.append({
            "user_id": uid,
            "gender": gender,
            "age": age,
            "weight_kg": weight,
            "height_cm": height,
            "activity_level": act,
            "dietary_goal": goal,
            "diet_type": diet,
            "food_allergy": allergy,
            "bmr": bmr,
            "tdee": tdee,
            "target_kcal": target_kcal,
            "target_protein_g": target_protein,
            "target_fat_g": target_fat,
            "target_carbs_g": max(target_carbs, 20),
            "meals_per_day": int(rng.choice([2, 3, 4, 5])),
            "snacks_per_day": int(rng.choice([0, 1, 2])),
            "meal_satisfaction_score": round(float(base_satisfaction), 2),
            "prefers_quick_meals": int(rng.random() > 0.5),
            "prefers_budget_meals": int(rng.random() > 0.4),
        })

    users = pd.DataFrame(rows)
    logger.info(f"Synthetic users generated: {users.shape}")
    return users


# ===========================================================================
# STEP 9 - Generate synthetic meal feedback
# ===========================================================================

def generate_synthetic_feedback(users: pd.DataFrame, n_feedback: int = 20000) -> pd.DataFrame:
    """
    Simulate meal feedback records - user x food interactions with ratings.
    Used by the feedback learning loop in Phase 6.
    """
    rng = np.random.default_rng(7)
    user_ids = users["user_id"].values
    # Use category ids as food proxies (real fdc_ids added during integration)
    category_ids = list(range(1, 29))
    meal_types = ["breakfast", "lunch", "dinner", "snack"]

    rows = []
    for i in range(n_feedback):
        uid = rng.choice(user_ids)
        user_row = users[users["user_id"] == uid].iloc[0]
        cat_id   = rng.choice(category_ids)
        meal_type = rng.choice(meal_types)

        # Rating influenced by goal alignment
        base = rng.normal(3.5, 0.8)
        # Prefer protein-rich for muscle_gain
        if user_row["dietary_goal"] == "muscle_gain" and cat_id in [5, 10, 13, 15]:
            base += 0.5
        if user_row["dietary_goal"] == "weight_loss" and cat_id in [9, 11]:
            base += 0.4
        rating = float(np.clip(base, 1.0, 5.0))

        rows.append({
            "feedback_id": i + 1,
            "user_id": int(uid),
            "food_category_id": cat_id,
            "meal_type": meal_type,
            "rating": round(rating, 2),
            "would_eat_again": int(rating >= 3.5),
            "portion_satisfied": int(rng.random() > 0.3),
            "timestamp": pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 730))),
        })

    feedback = pd.DataFrame(rows)
    logger.info(f"Synthetic feedback generated: {feedback.shape}")
    return feedback


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_preprocessing_pipeline() -> dict[str, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("PHASE 1 - Data Analysis & Preprocessing Pipeline")
    logger.info("=" * 60)

    tables   = load_raw_tables()
    matrix   = build_nutrition_matrix(tables)
    merged   = merge_food_metadata(matrix, tables)
    with_portions = add_portion_data(merged, tables)
    cleaned  = clean_and_impute(with_portions)
    featured = feature_engineering(cleaned)
    encoded, full = encode_and_normalise(featured)

    # Synthetic data
    users    = generate_synthetic_users(5000)
    feedback = generate_synthetic_feedback(users)

    # -- Save all processed files ------------------------------------------
    full.to_csv(PROCESSED_DIR / "nutrition_full.csv", index=False)
    encoded.to_csv(PROCESSED_DIR / "nutrition_encoded.csv", index=False)
    users.to_csv(PROCESSED_DIR / "users.csv", index=False)
    feedback.to_csv(PROCESSED_DIR / "feedback.csv", index=False)

    # -- Summary stats -----------------------------------------------------
    logger.info("\n PREPROCESSING SUMMARY")
    logger.info(f"  Foods (full):          {full.shape}")
    logger.info(f"  Foods (ML-encoded):    {encoded.shape}")
    logger.info(f"  Synthetic users:       {users.shape}")
    logger.info(f"  Synthetic feedback:    {feedback.shape}")
    logger.info(f"  Null values remaining: {full.isnull().sum().sum()}")
    logger.info(f"  Food categories:       {full['category_name'].nunique()}")
    logger.info(f"  Diet tags summary:")
    for tag in ["is_high_protein", "is_low_carb", "is_low_fat", "is_low_calorie", "is_high_fiber"]:
        if tag in full.columns:
            logger.info(f"    {tag}: {full[tag].sum()} foods")

    logger.info("[OK] PHASE 1 COMPLETE - all files saved to data/processed/")
    return {
        "nutrition_full": full,
        "nutrition_encoded": encoded,
        "users": users,
        "feedback": feedback,
    }


if __name__ == "__main__":
    run_preprocessing_pipeline()
