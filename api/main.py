"""
api/main.py  —  FastAPI Backend
Smart AI Diet & Meal Optimization System
"""

import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

# Force UTF-8 on Windows to prevent cp1252 codec crashes
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Resolve project root and add to sys.path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization.meal_plan_generator import MealPlanGenerator
from utils.grocery_generator import GroceryListGenerator
from utils.feedback_loop import FeedbackLearningLoop

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
FRONTEND_DIR  = ROOT / "frontend"


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_meal_gen:      Optional[MealPlanGenerator]    = None
_grocery_gen:   Optional[GroceryListGenerator] = None
_feedback_loop: Optional[FeedbackLearningLoop] = None
_nutrition_df:  Optional[pd.DataFrame]         = None
_user_store:    dict = {}


def get_meal_gen() -> MealPlanGenerator:
    global _meal_gen
    if _meal_gen is None:
        _meal_gen = MealPlanGenerator()
    return _meal_gen


def get_grocery_gen() -> GroceryListGenerator:
    global _grocery_gen
    if _grocery_gen is None:
        _grocery_gen = GroceryListGenerator()
    return _grocery_gen


def get_feedback() -> FeedbackLearningLoop:
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = FeedbackLearningLoop()
    return _feedback_loop


def get_nutrition() -> pd.DataFrame:
    global _nutrition_df
    if _nutrition_df is None:
        _nutrition_df = pd.read_csv(
            PROCESSED_DIR / "nutrition_full.csv", low_memory=False
        )
    return _nutrition_df


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    user_id:        int
    name:           str   = "User"
    age:            int
    weight_kg:      float
    height_cm:      float
    gender:         str   = "male"
    activity_level: str   = "moderate"
    dietary_goal:   str   = "maintenance"
    diet_type:      str   = "omnivore"
    food_allergy:   str   = "none"
    meals_per_day:  int   = 3
    snacks_per_day: int   = 1


class MealPlanRequest(BaseModel):
    user:       UserProfile
    days:       int           = 7
    start_date: Optional[str] = None


class FeedbackEvent(BaseModel):
    user_id:           int
    fdc_id:            Optional[int] = None
    food_category_id:  int
    meal_type:         str   = "lunch"
    rating:            float
    would_eat_again:   Optional[int] = None
    portion_satisfied: Optional[int] = None


# ---------------------------------------------------------------------------
# FastAPI app (created without lifespan first, added below)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Smart AI Diet & Meal Optimization API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — open for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Frontend static file routes
#
# The HTML uses bare relative paths (href="style.css", src="app.js").
# The browser therefore requests GET /style.css and GET /app.js from the
# API origin.  We register explicit routes for every asset so the API
# routes (/health, /docs, /nutrition/...) are never shadowed.
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_index():
    f = FRONTEND_DIR / "index.html"
    if not f.exists():
        raise HTTPException(404, f"index.html not found. Looked in: {FRONTEND_DIR}")
    return FileResponse(str(f), media_type="text/html")


@app.get("/app", include_in_schema=False)
def serve_app_page():
    f = FRONTEND_DIR / "index.html"
    if not f.exists():
        raise HTTPException(404, f"index.html not found. Looked in: {FRONTEND_DIR}")
    return FileResponse(str(f), media_type="text/html")


@app.get("/style.css", include_in_schema=False)
def serve_css():
    f = FRONTEND_DIR / "style.css"
    if not f.exists():
        raise HTTPException(404, f"style.css not found. Looked in: {FRONTEND_DIR}")
    return FileResponse(str(f), media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def serve_js():
    f = FRONTEND_DIR / "app.js"
    if not f.exists():
        raise HTTPException(404, f"app.js not found. Looked in: {FRONTEND_DIR}")
    return FileResponse(str(f), media_type="application/javascript")


# Mount /static for any additional assets (images, fonts, etc.)
if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static",
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "foods_loaded": len(get_nutrition()),
        "feedback_records": get_feedback().store.count(),
        "models": {
            "user_nutrition_net": (MODELS_DIR / "user_nutrition_net.pt").exists(),
            "food_scoring_net":   (MODELS_DIR / "food_scoring_net.pt").exists(),
            "mf_model":           (MODELS_DIR / "mf_model.pt").exists(),
        },
    }


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

@app.post("/users/register", status_code=201, tags=["Users"])
def register_user(profile: UserProfile):
    data    = profile.model_dump()
    targets = get_meal_gen().predictor.predict(data)
    data["predicted_targets"] = targets
    _user_store[profile.user_id] = data
    return {
        "message": "User registered",
        "user_id": profile.user_id,
        "targets": targets,
    }


@app.get("/users/{user_id}", tags=["Users"])
def get_user(user_id: int):
    if user_id not in _user_store:
        raise HTTPException(404, f"User {user_id} not found")
    prefs = get_feedback().get_user_preferences(user_id)
    return {"profile": _user_store[user_id], "preferences": prefs}


# ---------------------------------------------------------------------------
# Meal plans
# ---------------------------------------------------------------------------

@app.post("/meal-plan/generate", tags=["Meal Plans"])
def generate_meal_plan(req: MealPlanRequest):
    user  = req.user.model_dump()
    start = date.fromisoformat(req.start_date) if req.start_date else date.today()
    try:
        result = get_meal_gen().generate(user, days=req.days, start_date=start)
    except Exception as exc:
        logger.error("Plan generation error: %s", exc, exc_info=True)
        raise HTTPException(500, str(exc))
    return {
        "user_id": user["user_id"],
        "targets": result["targets"],
        "plan":    result["plan"],
    }


@app.get("/meal-plan/{user_id}", tags=["Meal Plans"])
def get_meal_plan(user_id: int):
    p = PROCESSED_DIR / f"meal_plan_user_{user_id}.json"
    if not p.exists():
        raise HTTPException(404, "No plan found. Call POST /meal-plan/generate first.")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@app.get("/meal-plan/{user_id}/report", tags=["Meal Plans"],
         response_class=PlainTextResponse)
def get_report(user_id: int):
    p = PROCESSED_DIR / f"meal_plan_user_{user_id}.txt"
    if not p.exists():
        raise HTTPException(404, "No text report found.")
    return p.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Grocery
# ---------------------------------------------------------------------------

@app.get("/grocery/{user_id}", tags=["Grocery"])
def get_grocery(user_id: int):
    plan_path = PROCESSED_DIR / f"meal_plan_user_{user_id}.json"
    if not plan_path.exists():
        raise HTTPException(404, "Generate a meal plan first.")
    result = get_grocery_gen().generate(plan_path)
    if not result:
        raise HTTPException(500, "Grocery generation failed.")
    return result["data"]


@app.get("/grocery/{user_id}/text", tags=["Grocery"],
         response_class=PlainTextResponse)
def get_grocery_text(user_id: int):
    txt = PROCESSED_DIR / f"grocery_user_{user_id}.txt"
    if not txt.exists():
        get_grocery(user_id)
    if txt.exists():
        return txt.read_text(encoding="utf-8")
    raise HTTPException(500, "Could not generate grocery list.")


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

@app.post("/feedback", status_code=201, tags=["Feedback"])
def submit_feedback(event: FeedbackEvent):
    get_feedback().submit_feedback(event.model_dump())
    return {"message": "Feedback recorded", "user_id": event.user_id}


@app.post("/feedback/learn", tags=["Feedback"])
def trigger_learning(min_records: int = Query(20, ge=5)):
    get_feedback().run_learning_cycle(min_new_records=min_records)
    return {
        "message":       "Learning cycle complete",
        "total_records": get_feedback().store.count(),
    }


# ---------------------------------------------------------------------------
# Nutrition
# ---------------------------------------------------------------------------

@app.get("/nutrition/search", tags=["Nutrition"])
def search_nutrition(
    q:        Optional[str] = Query(None, description="Food name keyword"),
    category: Optional[str] = Query(None, description="Category substring"),
    limit:    int            = Query(10, ge=1, le=50),
):
    df = get_nutrition().copy()
    if q:
        df = df[df["description"].str.contains(q, case=False, na=False)]
    if category:
        df = df[df["category_name"].str.contains(category, case=False, na=False)]
    cols = [
        "fdc_id", "description", "category_name",
        "energy_kcal", "protein_g", "fat_g", "carbs_g",
        "fiber_g", "serving_gram_weight",
    ]
    cols = [c for c in cols if c in df.columns]
    out  = df[cols].head(limit).fillna(0).round(2).to_dict(orient="records")
    return {"count": len(out), "results": out}


@app.get("/nutrition/categories/all", tags=["Nutrition"])
def list_categories():
    df   = get_nutrition()
    cats = (
        df.groupby("category_name")
        .agg(food_count=("fdc_id", "count"), avg_kcal=("energy_kcal", "mean"))
        .reset_index()
        .round(1)
    )
    return cats.to_dict(orient="records")


@app.get("/nutrition/{fdc_id}", tags=["Nutrition"])
def get_food(fdc_id: int):
    df  = get_nutrition()
    row = df[df["fdc_id"] == fdc_id]
    if row.empty:
        raise HTTPException(404, f"Food {fdc_id} not found")
    cols = [
        "fdc_id", "description", "category_name",
        "energy_kcal", "protein_g", "fat_g", "carbs_g",
        "fiber_g", "sodium_mg", "nutrient_density_score", "serving_gram_weight",
    ]
    cols = [c for c in cols if c in df.columns]
    return row[cols].fillna(0).round(3).to_dict(orient="records")[0]


# ---------------------------------------------------------------------------
# Startup / shutdown (lifespan)
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(application: FastAPI):
    # --- startup ---
    logger.info("Smart Diet API v2 starting...")
    logger.info("Project root:  %s", ROOT)
    logger.info("Frontend dir:  %s  (exists=%s)", FRONTEND_DIR, FRONTEND_DIR.exists())
    try:
        get_nutrition()
        get_meal_gen()
        get_feedback()
        logger.info("All services ready. Foods loaded: %d", len(get_nutrition()))
    except Exception as exc:
        logger.error("Startup error: %s", exc)
    yield
    # --- shutdown (nothing to clean up) ---


app.router.lifespan_context = lifespan

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
