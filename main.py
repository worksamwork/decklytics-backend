import os
import io
import base64
from typing import Optional

import asyncpg
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# -------- App & CORS --------
app = FastAPI(title="Deckticks API")
DB_URL = os.environ.get("DATABASE_URL", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Helpers --------
def z(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 7:
        return 0.0
    return float((s.iloc[-1] - s.mean()) / (s.std() + 1e-9))

def to_score(zv: float) -> float:
    # 80 is neutral; worse goes down, better goes up
    return float(np.clip(80.0 - 20.0 * zv, 0.0, 100.0))

def parse_ga4(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "date" not in df:
        raise ValueError("GA4 needs 'date' column")
    for col in ["sessions", "users", "conversions", "cost"]:
        if col not in df:
            df[col] = 0
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

def parse_clarity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if "date" not in df:
        raise ValueError("Clarity needs 'date' column")
    for col in ["page", "rage_clicks", "dead_clicks", "js_errors", "sessions"]:
        if col not in df:
            df[col] = 0
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

def compute(day_ga4: pd.DataFrame, day_cl: pd.DataFrame):
    g = (
        day_ga4.groupby("date")
        .agg({"sessions": "sum", "users": "sum", "conversions": "sum", "cost": "sum"})
        .reset_index()
    )
    c = (
        day_cl.groupby("date")
        .agg({"rage_clicks": "sum", "dead_clicks": "sum", "js_errors": "sum", "sessions": "sum"})
        .reset_index()
    )
    day = pd.merge(g, c, on="date", how="outer").fillna(0.0).sort_values("date")

    # sessions_x = GA4 sessions; sessions_y = Clarity sessions after merge
    day["cvr"] = np.where(day.get("sessions_x", 0) > 0, day["conversions"] / day["sessions_x"], 0.0)

    cac_series = day["cost"] / day["conversions"].replace(0, np.nan)
    rage_rate = np.where(day.get("sessions_y", 0) > 0, day["rage_clicks"] / day["sessions_y"], 0.0)

    scores = {
        "ux": to_score(z(pd.Series(rage_rate))),
        "acquisition": to_score(z(cac_series.fillna(method="ffill").fillna(cac_series.mean()))),
        "activation": to_score(-z(day["cvr"])),
        "retention": 80.0,
        "revenue": 80.0,
    }
    return day, scores

def b64_plot(x, y, title):
    import matplotlib.pyplot as plt  # imported here so startup is lighter
    fig = plt.figure(figsize=(6, 3))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Date")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# In-memory store for the MVP
STATE = {}  # { project_id: { "ga4": DataFrame, "clarity": DataFrame } }

# -------- Routes --------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/upload")
async def upload(
    project_id: str = Form(...),
    ga4: Optional[UploadFile] = File(None),
    clarity: Optional[UploadFile] = File(None),
):
    if ga4 is None and clarity is None:
        return JSONResponse({"error": "upload at least one file"}, status_code=400)

    gdf = pd.DataFrame()
    cdf = pd.DataFrame()

    if ga4:
        g_bytes = await ga4.read()
        gdf = parse_ga4(pd.read_csv(io.BytesIO(g_bytes)))
    if clarity:
        c_bytes = await clarity.read()
        cdf = parse_clarity(pd.read_csv(io.BytesIO(c_bytes)))

    STATE[project_id] = {"ga4": gdf, "clarity": cdf}
    return {"ok": True, "rows": {"ga4": int(len(gdf)), "clarity": int(len(cdf))}}

@app.get("/api/dashboard")
async def dashboard(project_id: str):
    data = STATE.get(project_id)
    if not data:
        return JSONResponse({"error": "no data uploaded yet"}, status_code=404)

    g = data.get("ga4", pd.DataFrame(columns=["date"]))
    c = data.get("clarity", pd.DataFrame(columns=["date"]))
    day, scores = compute(g, c)

    dates = pd.to_datetime(day["date"], errors="coerce").fillna(method="ffill")

    # Build charts
    rage_vals = day["rage_clicks"] if "rage_clicks" in day else pd.Series([0] * len(day))
    rage_png = b64_plot(dates, rage_vals, "Rage clicks")

    cac_raw = day["cost"] / day["conversions"].replace(0, np.nan)
    cac_png = b64_plot(dates, cac_raw.fillna(method="ffill").fillna(0), "CAC")

    cvr_vals = day["cvr"] if "cvr" in day else pd.Series([0] * len(day))
    cvr_png = b64_plot(dates, cvr_vals, "Conversion rate")

    fixes = []
    if scores["ux"] < 70:
        fixes.append({
            "pillar": "ux",
            "title": "Rage clicks rising",
            "why": "Above baseline z-score",
            "action": "Increase tap targets and fix JS errors on top pages",
            "impact": "H",
            "effort": "S",
        })
    if scores["acquisition"] < 70:
        fixes.append({
            "pillar": "acquisition",
            "title": "CAC worsening",
            "why": "Higher than baseline",
            "action": "Pause non-converting ad sets and tighten audiences",
            "impact": "H",
            "effort": "M",
        })
    if scores["activation"] < 70:
        fixes.append({
            "pillar": "activation",
            "title": "Signup to Aha down",
            "why": "CVR dropped vs prior week",
            "action": "Shorten form and add first-run guide",
            "impact": "M",
            "effort": "S",
        })
    if not fixes:
        fixes.append({
            "pillar": "overview",
            "title": "Steady week",
            "why": "Near baseline across pillars",
            "action": "Run a 7-day CTA experiment on the hero",
            "impact": "M",
            "effort": "S",
        })

    return {
        "scores": scores,
        "fixes": fixes,
        "charts": {
            "rage": "data:image/png;base64," + rage_png,
            "cac": "data:image/png;base64," + cac_png,
            "cvr": "data:image/png;base64," + cvr_png,
        },
    }

# Optional: DB connectivity smoke test (not used in MVP paths)
@app.get("/db/ping")
async def db_ping():
    if not DB_URL:
        return {"ok": False, "reason": "no DATABASE_URL env"}
    conn = await asyncpg.connect(DB_URL)
    v = await conn.fetchval("select version()")
    await conn.close()
    return {"ok": True, "version": v}
