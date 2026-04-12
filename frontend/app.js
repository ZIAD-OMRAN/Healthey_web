/* ═══════════════════════════════════════════════════════════════
   NutriAI — Frontend App Logic
   Connects to FastAPI backend at localhost:8000
   ═══════════════════════════════════════════════════════════════ */

const API = "http://localhost:8000";
let currentUserId = null;
let currentPlan   = null;

// ── Utility ───────────────────────────────────────────────────
const qs  = (sel, ctx = document) => ctx.querySelector(sel);
const all = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

function fmt(n, decimals = 0) {
  if (n == null || isNaN(n)) return "—";
  return Number(n).toFixed(decimals);
}
function cap(s) {
  return String(s).replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

// ── Health check on load ───────────────────────────────────────
async function checkHealth() {
  const badge = qs("#status-badge");
  try {
    const h = await apiFetch("/health");
    badge.className = "status-badge ok";
    badge.innerHTML = `<span class="dot"></span> ${h.foods_loaded.toLocaleString()} foods loaded`;
  } catch {
    badge.className = "status-badge fail";
    badge.innerHTML = `<span class="dot"></span> API offline`;
  }
}

// ── Navigation ─────────────────────────────────────────────────
all(".nav-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    all(".nav-tab").forEach(b => b.classList.remove("active"));
    all(".section").forEach(s => s.classList.remove("active"));
    btn.classList.add("active");
    qs(`#section-${btn.dataset.section}`).classList.add("active");
  });
});

// ═══════════════════════════════════════════════════════════════
// PLANNER
// ═══════════════════════════════════════════════════════════════

qs("#profile-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = qs("#gen-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px;display:inline-block"></span> Generating…';

  const user = buildUserPayload();
  currentUserId = user.user_id;

  // Hide previous output
  qs("#plan-output").style.display    = "none";
  qs("#grocery-output").style.display = "none";
  qs("#targets-card").style.display   = "none";
  qs("#spinner").style.display        = "flex";

  try {
    // 1. Register user → get predicted targets
    const reg = await apiFetch("/users/register", {
      method: "POST", body: JSON.stringify(user),
    });
    renderTargets(reg.targets);

    // 2. Generate meal plan
    const planRes = await apiFetch("/meal-plan/generate", {
      method: "POST",
      body: JSON.stringify({
        user, days: parseInt(qs("#f-days").value),
      }),
    });
    currentPlan = planRes;
    renderPlan(planRes.plan);

  } catch (err) {
    qs("#spinner").style.display = "none";
    showPlanError(err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">✦</span> Generate Meal Plan';
    qs("#spinner").style.display = "none";
  }
});

function buildUserPayload() {
  return {
    user_id:        parseInt(qs("#f-uid").value),
    name:           qs("#f-name").value || "User",
    age:            parseInt(qs("#f-age").value),
    weight_kg:      parseFloat(qs("#f-weight").value),
    height_cm:      parseFloat(qs("#f-height").value),
    gender:         qs("#f-gender").value,
    activity_level: qs("#f-activity").value,
    dietary_goal:   qs("#f-goal").value,
    diet_type:      qs("#f-diet").value,
    food_allergy:   qs("#f-allergy").value,
    meals_per_day:  parseInt(qs("#f-meals").value),
    snacks_per_day: 1,
  };
}

function renderTargets(t) {
  qs("#t-kcal").textContent  = fmt(t.target_kcal);
  qs("#t-prot").textContent  = fmt(t.target_protein_g);
  qs("#t-fat").textContent   = fmt(t.target_fat_g);
  qs("#t-carbs").textContent = fmt(t.target_carbs_g);
  qs("#targets-card").style.display = "block";
}

function renderPlan(plan) {
  const container = qs("#plan-days");
  container.innerHTML = "";

  plan.forEach((day, i) => {
    const card = document.createElement("div");
    card.className = "day-card";

    const t = day.totals || {};
    const g = day.targets || {};

    card.innerHTML = `
      <div class="day-header" onclick="toggleDay(this)">
        <span class="day-name">📅 ${day.day} <small style="color:var(--text-muted);font-weight:400;font-size:0.75rem">${day.date}</small></span>
        <div class="day-totals">
          <span class="day-total-pill" style="color:#f97316">${fmt(t.energy_kcal)} kcal</span>
          <span class="day-total-pill" style="color:#3b82f6">P ${fmt(t.protein_g)}g</span>
          <span class="day-total-pill" style="color:#a78bfa">F ${fmt(t.fat_g)}g</span>
          <span class="day-total-pill" style="color:#34d399">C ${fmt(t.carbs_g)}g</span>
        </div>
        <span class="day-toggle">${i === 0 ? "▲" : "▼"}</span>
      </div>
      <div class="day-body" style="display:${i === 0 ? "block" : "none"}">
        ${renderMeals(day.meals)}
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--border);
                    display:flex;gap:8px;flex-wrap:wrap;font-size:0.72rem;color:var(--text-muted)">
          <span>Targets → ${fmt(g.energy_kcal)} kcal · P ${fmt(g.protein_g)}g · F ${fmt(g.fat_g)}g · C ${fmt(g.carbs_g)}g</span>
        </div>
      </div>`;
    container.appendChild(card);
  });

  qs("#plan-output").style.display = "block";
}

function renderMeals(meals) {
  if (!meals || !meals.length) return '<p class="empty-state">No meals</p>';
  return meals.map(meal => `
    <div class="meal-block">
      <div class="meal-slot-label">
        ${slotEmoji(meal.slot)} ${cap(meal.slot)} — ${fmt(meal.totals?.energy_kcal)} kcal
      </div>
      ${(meal.foods || []).map(f => `
        <div class="food-row">
          <span class="food-name">${f.name}</span>
          <div class="food-meta">
            <span>${fmt(f.serving_g)}g</span>
            <span class="kcal">${fmt(f.energy_kcal)} kcal</span>
            <span>P ${fmt(f.protein_g)}g</span>
          </div>
        </div>`).join("")}
    </div>`).join("");
}

function slotEmoji(slot) {
  if (slot === "breakfast") return "🌅";
  if (slot === "lunch")     return "☀️";
  if (slot === "dinner")    return "🌙";
  return "🍎";
}

window.toggleDay = (header) => {
  const body   = header.nextElementSibling;
  const toggle = qs(".day-toggle", header);
  const open   = body.style.display !== "none";
  body.style.display = open ? "none" : "block";
  toggle.textContent = open ? "▼" : "▲";
};

function showPlanError(msg) {
  qs("#plan-days").innerHTML = `<p class="error-state">⚠ ${msg}</p>`;
  qs("#plan-output").style.display = "block";
}

// ── Grocery ────────────────────────────────────────────────────
qs("#grocery-btn").addEventListener("click", async () => {
  if (!currentUserId) return;
  const btn = qs("#grocery-btn");
  btn.textContent = "Loading…";
  btn.disabled    = true;
  try {
    const data = await apiFetch(`/grocery/${currentUserId}`);
    renderGrocery(data);
  } catch (err) {
    alert("Grocery error: " + err.message);
  } finally {
    btn.textContent = "🛒 Get Grocery List";
    btn.disabled    = false;
  }
});

function renderGrocery(data) {
  qs("#grocery-cost").textContent = `Est. $${fmt(data.estimated_cost_usd, 2)}/week`;

  const container = qs("#grocery-categories");
  container.innerHTML = "";

  const cats = data.by_category || {};
  Object.entries(cats).sort(([a],[b]) => a.localeCompare(b)).forEach(([cat, items]) => {
    const block = document.createElement("div");
    block.className = "grocery-cat-block";
    block.innerHTML = `
      <div class="grocery-cat-title">📦 ${cat}</div>
      ${items.map(item => `
        <div class="grocery-item">
          <span class="grocery-item-name">${item.name?.slice(0,50) || "?"}</span>
          <div class="grocery-item-right">
            <span>${item.purchase_grams}g</span>
            <span class="cost">$${fmt(item.estimated_cost_usd,2)}</span>
          </div>
        </div>`).join("")}`;
    container.appendChild(block);
  });
  qs("#grocery-output").style.display = "block";
  qs("#grocery-output").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ═══════════════════════════════════════════════════════════════
// FOOD SEARCH
// ═══════════════════════════════════════════════════════════════

qs("#search-btn").addEventListener("click", doSearch);
qs("#search-input").addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });

async function doSearch() {
  const q   = qs("#search-input").value.trim();
  const btn = qs("#search-btn");
  if (!q) return;

  btn.textContent = "…";
  btn.disabled    = true;

  try {
    const data = await apiFetch(`/nutrition/search?q=${encodeURIComponent(q)}&limit=20`);
    renderSearchResults(data.results);
  } catch (err) {
    qs("#search-results").innerHTML = `<p class="error-state">⚠ ${err.message}</p>`;
  } finally {
    btn.textContent = "Search";
    btn.disabled    = false;
  }
}

function renderSearchResults(results) {
  const container = qs("#search-results");
  if (!results.length) {
    container.innerHTML = `<div class="empty-state"><div class="big">🔍</div>No results found.</div>`;
    return;
  }
  container.innerHTML = results.map(f => `
    <div class="food-card">
      <div class="food-card-name">${f.description || "?"}</div>
      <div class="food-card-cat">${f.category_name || "—"}</div>
      <div class="food-macros">
        <span class="macro-pill" style="color:#f97316;border-color:rgba(249,115,22,0.3)">${fmt(f.energy_kcal)} kcal</span>
        <span class="macro-pill" style="color:#3b82f6;border-color:rgba(59,130,246,0.3)">P ${fmt(f.protein_g)}g</span>
        <span class="macro-pill" style="color:#a78bfa;border-color:rgba(167,139,250,0.3)">F ${fmt(f.fat_g)}g</span>
        <span class="macro-pill" style="color:#34d399;border-color:rgba(52,211,153,0.3)">C ${fmt(f.carbs_g)}g</span>
        <span class="macro-pill">${fmt(f.serving_gram_weight)}g/serving</span>
      </div>
    </div>`).join("");
}

// ═══════════════════════════════════════════════════════════════
// FEEDBACK
// ═══════════════════════════════════════════════════════════════

let selectedRating = 0;

all(".star").forEach(star => {
  star.addEventListener("click", () => {
    selectedRating = parseInt(star.dataset.v);
    qs("#fb-rating").value = selectedRating;
    all(".star").forEach((s, i) => {
      s.classList.toggle("active", i < selectedRating);
    });
    qs("#fb-btn").disabled = false;
  });
  star.addEventListener("mouseenter", () => {
    const v = parseInt(star.dataset.v);
    all(".star").forEach((s, i) => s.classList.toggle("active", i < v));
  });
  star.addEventListener("mouseleave", () => {
    all(".star").forEach((s, i) => s.classList.toggle("active", i < selectedRating));
  });
});

qs("#feedback-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const rating = parseFloat(qs("#fb-rating").value);
  if (!rating) return;

  const btn = qs("#fb-btn");
  btn.disabled    = true;
  btn.textContent = "Submitting…";

  const msg = qs("#feedback-msg");
  msg.style.display = "none";

  try {
    await apiFetch("/feedback", {
      method: "POST",
      body: JSON.stringify({
        user_id:          parseInt(qs("#fb-uid").value),
        food_category_id: parseInt(qs("#fb-cat").value),
        meal_type:        qs("#fb-meal").value,
        rating,
        would_eat_again:  rating >= 3.5 ? 1 : 0,
      }),
    });
    msg.className   = "feedback-msg success";
    msg.textContent = `✓ Rating of ${rating}/5 recorded! The AI will learn from your feedback.`;
    msg.style.display = "block";
    // Reset stars
    selectedRating = 0;
    all(".star").forEach(s => s.classList.remove("active"));
    qs("#fb-rating").value = 0;
  } catch (err) {
    msg.className   = "feedback-msg error";
    msg.textContent = "⚠ " + err.message;
    msg.style.display = "block";
  } finally {
    btn.disabled    = false;
    btn.textContent = "Submit Rating";
  }
});

// ── Init ───────────────────────────────────────────────────────
checkHealth();
setInterval(checkHealth, 30000);  // re-check every 30s
