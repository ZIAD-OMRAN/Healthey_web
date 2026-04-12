"""
main.py - Master Entry Point
Smart AI Diet & Meal Optimization System

Usage:
  python main.py pipeline   - full data + training pipeline
  python main.py demo       - generate plans for 3 demo users
  python main.py api        - start FastAPI on port 8000
  python main.py test       - run integration tests
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows to prevent cp1252 codec crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is always on sys.path
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Create logs directory before setting up file handler
(ROOT / "logs").mkdir(exist_ok=True)

# Configure logging - ASCII only, UTF-8 file output
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_stream = logging.StreamHandler(sys.stdout)
_stream.setFormatter(_fmt)

_file = logging.FileHandler(ROOT / "logs" / "main.log", encoding="utf-8")
_file.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_stream, _file])
logger = logging.getLogger(__name__)

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def run_full_pipeline():
    t0 = time.time()

    logger.info("PHASE 1 - Data Preprocessing")
    from utils.data_preprocessing import run_preprocessing_pipeline
    run_preprocessing_pipeline()

    logger.info("PHASE 2 - Neural Network Training")
    from models.neural_networks import run_model_training
    run_model_training()

    logger.info("PHASE 3 - Optimization Demo")
    from optimization.meal_optimizer import run_optimization_demo
    run_optimization_demo()

    logger.info("PHASE 4+5 - Meal Plans + Grocery")
    from optimization.meal_plan_generator import run_meal_plan_demo
    from utils.grocery_generator import run_grocery_demo
    run_meal_plan_demo()
    run_grocery_demo()

    logger.info("PHASE 6 - Feedback Loop")
    from utils.feedback_loop import run_feedback_demo
    run_feedback_demo()

    logger.info("Full pipeline complete in %.1fs", time.time() - t0)


# ---------------------------------------------------------------------------
# DEMO
# ---------------------------------------------------------------------------

def run_demo():
    from optimization.meal_plan_generator import MealPlanGenerator
    from utils.grocery_generator import GroceryListGenerator

    gen     = MealPlanGenerator()
    grocery = GroceryListGenerator()

    demo_users = [
        {
            "user_id": 101, "name": "Ali (Weight Loss)", "age": 32,
            "weight_kg": 92.0, "height_cm": 180.0, "gender": "male",
            "activity_level": "moderate", "dietary_goal": "weight_loss",
            "diet_type": "omnivore", "food_allergy": "none",
            "meals_per_day": 3, "snacks_per_day": 1,
        },
        {
            "user_id": 102, "name": "Nour (Vegan Muscle)", "age": 24,
            "weight_kg": 58.0, "height_cm": 162.0, "gender": "female",
            "activity_level": "active", "dietary_goal": "muscle_gain",
            "diet_type": "vegan", "food_allergy": "none",
            "meals_per_day": 4, "snacks_per_day": 2,
        },
        {
            "user_id": 103, "name": "Hassan (Heart / Keto)", "age": 52,
            "weight_kg": 80.0, "height_cm": 173.0, "gender": "male",
            "activity_level": "light", "dietary_goal": "heart_health",
            "diet_type": "keto", "food_allergy": "gluten",
            "meals_per_day": 3, "snacks_per_day": 0,
        },
    ]

    for user in demo_users:
        logger.info("Generating plan for: %s", user["name"])
        gen.generate(user, days=7)
        plan_path = PROCESSED_DIR / f"meal_plan_user_{user['user_id']}.json"
        if plan_path.exists():
            grocery.generate(plan_path)

    logger.info("Demo complete - plans saved to data/processed/")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def run_api():
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    logger.info("Starting Smart Diet API server...")
    logger.info("  Frontend -> http://localhost:8000/")
    logger.info("  API Docs -> http://localhost:8000/docs")
    logger.info("  Health   -> http://localhost:8000/health")

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

def run_tests():
    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            logger.info("  PASS  %s", name)
            passed += 1
        else:
            logger.error("  FAIL  %s  %s", name, detail)
            failed += 1

    import pandas as pd

    # File checks
    check("nutrition_full.csv",         (PROCESSED_DIR / "nutrition_full.csv").exists())
    check("users.csv",                  (PROCESSED_DIR / "users.csv").exists())
    check("feedback.csv",               (PROCESSED_DIR / "feedback.csv").exists())
    check("user_nutrition_net.pt",      (MODELS_DIR / "user_nutrition_net.pt").exists())
    check("food_scoring_net.pt",        (MODELS_DIR / "food_scoring_net.pt").exists())
    check("mf_model.pt",                (MODELS_DIR / "mf_model.pt").exists())
    check("user_feat_scaler.pkl",       (MODELS_DIR / "user_feat_scaler.pkl").exists())
    check("frontend/index.html",        (ROOT / "frontend" / "index.html").exists())
    check("frontend/style.css",         (ROOT / "frontend" / "style.css").exists())
    check("frontend/app.js",            (ROOT / "frontend" / "app.js").exists())

    # Data checks
    df = pd.read_csv(PROCESSED_DIR / "nutrition_full.csv", low_memory=False)
    check(">=20k food records",         len(df) >= 20000, f"got {len(df)}")
    check("energy_kcal column",         "energy_kcal" in df.columns)
    check("no nulls in energy_kcal",    df["energy_kcal"].isnull().sum() == 0)

    # ML checks
    try:
        from optimization.meal_plan_generator import UserTargetPredictor
        out = UserTargetPredictor().predict({
            "age": 30, "weight_kg": 70, "height_cm": 175, "gender": "male",
            "activity_level": "moderate", "dietary_goal": "maintenance",
            "diet_type": "omnivore", "food_allergy": "none",
            "meals_per_day": 3, "snacks_per_day": 1,
        })
        check("NN prediction runs",     isinstance(out["target_kcal"], float))
        check("Predicted kcal 800-5000", 800 <= out["target_kcal"] <= 5000,
              f"{out['target_kcal']:.0f}")
    except Exception as exc:
        check("NN prediction runs",     False, str(exc))

    # Optimizer check
    try:
        from optimization.meal_optimizer import optimize_daily_plan
        plan = optimize_daily_plan(
            df,
            {"target_kcal": 2000, "target_protein_g": 120,
             "target_fat_g": 65, "target_carbs_g": 240},
            meals_per_day=3, snacks_per_day=0,
        )
        check("LP optimizer runs",      len(plan) > 0)
    except Exception as exc:
        check("LP optimizer runs",      False, str(exc))

    # API checks
    try:
        from api.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        check("GET / -> 200",           client.get("/").status_code == 200)
        check("GET /app -> 200",        client.get("/app").status_code == 200)
        check("GET /health -> 200",     client.get("/health").status_code == 200)
        check("GET /nutrition/search",  client.get(
            "/nutrition/search?q=beef&limit=2").status_code == 200)
        check("POST /feedback -> 201",  client.post(
            "/feedback", json={"user_id": 1, "food_category_id": 5,
                               "meal_type": "lunch", "rating": 4.0}
        ).status_code == 201)
        check("GET /nutrition/categories", client.get(
            "/nutrition/categories/all").status_code == 200)
    except Exception as exc:
        check("API tests",              False, str(exc))

    logger.info("")
    logger.info("Tests passed: %d  |  Failed: %d", passed, failed)
    if failed == 0:
        logger.info("ALL TESTS PASSED")
    else:
        logger.warning("%d test(s) failed", failed)
    return failed == 0


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart AI Diet & Meal Optimization System")
    parser.add_argument(
        "mode",
        nargs="?",
        default="api",
        choices=["pipeline", "demo", "api", "test"],
        help="Run mode (default: api)",
    )
    args = parser.parse_args()

    {
        "pipeline": run_full_pipeline,
        "demo":     run_demo,
        "api":      run_api,
        "test":     run_tests,
    }[args.mode]()
