import os, io, base64, asyncpg, pandas as pd, numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Deckticks API")
DB_URL = os.environ.get("DATABASE_URL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def z(series: pd.Series):
    s = series.dropna()
    if len(s) < 7: return 0.0
    return float((s.iloc[-1] - s.mean()) / (s.std() + 1e-9))

def to_score(zv: float):
    return float(np.clip(80 - 20*zv, 0, 100))

def parse_ga4(df: pd.DataFrame):
    df = df.rename(columns={c:c.lower() for c in df.columns})
    if "date" not in df: raise ValueError("GA4 needs 'date' column")
    for col in ["sessions","users","conversions","cost"]:
        if col not in df: df[col] = 0
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

def parse_clarity(df: pd.DataFrame):
    df = df.rename(columns={c:c.lower() for c in df.columns})
    if "date" not in df: raise ValueError("Clarity needs 'date' column")
    for col in ["page","rage_clicks","dead_clicks","js_errors","sessions"]:
        if col not in df: df[col] = 0
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

def compute(day_ga4: pd.DataFrame, day_cl: pd.DataFrame):
    g = day_ga4.groupby("date").agg({"sessions":"sum","users":"sum","conversions":"sum","cost":"sum"}).reset_index()
    c = day_cl.groupby("date").agg({"rage_clicks":"sum","dead_clicks":"sum","js_errors":"sum","sessions":"sum"}).reset_index()
    day = pd.merge(g, c, on="date", how="outer").fillna(0).sort_values("date")

    day["cvr"] = np.where(day["sessions_x"]>0, day["conversions"]/day["sessions_x"], 0.0)
    cac_series = day["cost"] / day["conversions"].replace(0,np.nan)
    rage_rate = np.where(day["sessions_y"]>0, day["rage_clicks"]/day["sessions_y"], 0.0)

    scores = {
        "ux": to_score(z(pd.Series(rage_rate))),
        "acquisition": to_score(z(cac_series.fillna(method="ffill").fillna(cac_series.mean()))),
        "activation": to_score(-z(day["cvr"])),
        "retention": 80.0,
        "revenue": 80.0
    }
    return day, scores

def b64_plot(x, y, title):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6,3))
    plt.plot(x, y); plt.title(title); plt.xlabel("Date"); plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

STATE = {}  # in-memory, keyed by project_id (for MVP)

@app.get("/health")
def health(): return {"ok": True}

@app.post("/api/upload")
async def upload(project_id: str = Form(...), ga4: UploadFile | None = File(None), clarity: UploadFile | None = File(None)):
    if ga4 is None and clarity is None:
        return JSONResponse({"error":"upload at least one file"}, status_code=400)
    gdf = pd.DataFrame(); cdf = pd.DataFrame()
    if ga4:  gdf = parse_ga4(pd.read_csv(io.BytesIO(await ga4.read())))
    if clarity: cdf = parse_clarity(pd.read_csv(io.BytesIO(await clarity.read())))
    STATE[project_id] = {"ga4": gdf, "clarity": cdf}
    return {"ok": True, "rows":{"ga4": len(gdf), "clarity": len(cdf)}}

@app.get("/api/dashboard")
async def dashboard(project_id: str):
    data = STATE.get(project_id)
    if not data: return JSONResponse({"error":"no data uploaded yet"}, status_code=404)
    g = data.get("ga4", pd.DataFrame(columns=["date"]))
    c = data.get("clarity", pd.DataFrame(columns=["date"]))
    day, scores = compute(g, c)

    dates = pd.to_datetime(day["date"])
    rage_png = b64_plot(dates, day.get("rage_clicks", pd.Series([0]*len(day))), "Rage clicks")
    cac = day["cost"] / day["conversions"].replace(0,np.nan)
    import pandas as _pd
    cac_png = b64_plot(dates, cac.fillna(method="ffill").fillna(0), "CAC")
    cvr_png = b64_plot(dates, day.get("cvr", _pd.Series([0]*len(day))), "Conversion rate")

    fixes = []
    if scores["ux"] < 70: fixes.append({"pillar":"ux","title":"Rage clicks up","why":"Above baseline","action":"Fix tap targets/JS errors","impact":"H","effort":"S"})
    if scores["acquisition"] < 70: fixes.append({"pillar":"acquisition","title":"CAC worsening","why":"Higher than baseline","action":"Pause non-converting adsets","impact":"H","effort":"M"})
    if scores["activation"] < 70: fixes.append({"pillar":"activation","title":"Signup→Aha down","why":"CVR dropped","action":"Shorten form + guide first-run","impact":"M","effort":"S"})
    if not fixes: fixes.append({"pillar":"overview","title":"Steady week","why":"Near baseline","action":"Run 7-day CTA test","impact":"M","effort":"S"})

    return {
      "scores": scores,
      "fixes": fixes,
      "charts": {
        "rage": "data:image/png;base64,"+rage_png,
        "cac": "data:image/png
