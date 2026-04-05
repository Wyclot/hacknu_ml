from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from fastapi.responses import HTMLResponse
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = Path(os.getenv("CSV_PATH", BASE_DIR / "shap_test_top_reasons_per_user.csv"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.3"))

app = FastAPI(title="AI Churn Analyst API", version="0.1.0")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

REQUIRED_COLUMNS = {
    "user_id",
    "probability_vol_churn",
    "feature",
    "feature_value",
    "shap_value",
    "abs_shap",
}


def load_data() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["user_id"] = df["user_id"].astype(str)
    df["probability_vol_churn"] = pd.to_numeric(df["probability_vol_churn"], errors="coerce")
    df["shap_value"] = pd.to_numeric(df["shap_value"], errors="coerce")
    df["abs_shap"] = pd.to_numeric(df["abs_shap"], errors="coerce")

    return df


def get_df() -> pd.DataFrame:
    return load_data()


def user_frame(user_id: str) -> pd.DataFrame:
    df = get_df()
    user_df = df[df["user_id"] == str(user_id)].copy()

    if user_df.empty:
        raise HTTPException(status_code=404, detail="User not found")

    return user_df.sort_values("abs_shap", ascending=False)


def clean_feature_value(value: object) -> str:
    if pd.isna(value):
        return "missing"
    return str(value)


def summarize_reason(row: pd.Series) -> dict:
    direction = "increases risk" if row["shap_value"] > 0 else "reduces risk"
    return {
        "feature": row["feature"],
        "feature_value": clean_feature_value(row["feature_value"]),
        "shap_value": float(row["shap_value"]),
        "abs_shap": float(row["abs_shap"]),
        "direction": direction,
    }


def build_prompt(user_id: str, proba: float, reasons: list[dict]) -> str:
    lines = []
    for r in reasons:
        arrow = "raises churn risk" if r["shap_value"] > 0 else "lowers churn risk"
        lines.append(
            f"- {r['feature']} = {r['feature_value']} -> {arrow} (impact: {r['shap_value']:+.3f})"
        )

    features_text = "\n".join(lines)

    return f"""You are a product analyst.
User ID: {user_id}
Voluntary churn probability: {proba:.1%}

Model signals:
{features_text}

Write 2 short paragraphs for a Product Manager:
1. Why this user is likely to voluntarily churn
2. What action should be taken next

Rules:
- Use plain business language
- Do not mention SHAP, model internals, or technical ML terms
- Be concrete and practical
- Keep it concise"""


class UserSummary(BaseModel):
    user_id: str
    churn_probability: float
    top_positive_driver: Optional[str]
    top_negative_driver: Optional[str]


class UserDetail(BaseModel):
    user_id: str
    churn_probability: float
    predicted_label: str
    top_reasons: list[dict]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "csv_path": str(CSV_PATH),
        "threshold": THRESHOLD,
    }


@app.get("/stats")
def stats() -> dict:
    df = get_df()
    per_user = df.groupby("user_id", as_index=False)["probability_vol_churn"].first()
    avg_risk = float(per_user["probability_vol_churn"].mean()) if not per_user.empty else 0.0
    high_risk = int((per_user["probability_vol_churn"] >= THRESHOLD).sum())

    return {
        "users": int(per_user["user_id"].nunique()),
        "avg_risk": avg_risk,
        "high_risk_users": high_risk,
        "top_features": (
            df.groupby("feature", as_index=False)["abs_shap"]
            .mean()
            .sort_values("abs_shap", ascending=False)
            .head(8)
            .to_dict(orient="records")
        ),
    }


@app.get("/users", response_model=list[UserSummary])
def list_users(limit: int = Query(default=20, ge=1, le=200)) -> list[UserSummary]:
    df = get_df()
    top_by_user = df.groupby("user_id", as_index=False)["probability_vol_churn"].first()
    top_by_user = top_by_user.sort_values("probability_vol_churn", ascending=False).head(limit)

    items: list[UserSummary] = []

    for _, row in top_by_user.iterrows():
        user_id = str(row["user_id"])
        user_df = df[df["user_id"] == user_id].sort_values("abs_shap", ascending=False)

        positive = user_df[user_df["shap_value"] > 0]["feature"].head(1)
        negative = user_df[user_df["shap_value"] < 0]["feature"].head(1)

        items.append(
            UserSummary(
                user_id=user_id,
                churn_probability=float(row["probability_vol_churn"]),
                top_positive_driver=None if positive.empty else str(positive.iloc[0]),
                top_negative_driver=None if negative.empty else str(negative.iloc[0]),
            )
        )

    return items


@app.get("/users/{user_id}", response_model=UserDetail)
def get_user(user_id: str, top_n: int = Query(default=8, ge=3, le=20)) -> UserDetail:
    user_df = user_frame(user_id)
    proba = float(user_df["probability_vol_churn"].iloc[0])
    reasons = [summarize_reason(row) for _, row in user_df.head(top_n).iterrows()]

    return UserDetail(
        user_id=str(user_id),
        churn_probability=proba,
        predicted_label="vol_churn" if proba >= THRESHOLD else "not_churned",
        top_reasons=reasons,
    )


@app.get("/users/{user_id}/explanation")
def get_user_explanation(user_id: str, top_n: int = Query(default=8, ge=3, le=20)) -> dict:
    user_df = user_frame(user_id)
    proba = float(user_df["probability_vol_churn"].iloc[0])
    reasons = [summarize_reason(row) for _, row in user_df.head(top_n).iterrows()]

    if client is None:
        return {
            "user_id": str(user_id),
            "churn_probability": proba,
            "predicted_label": "vol_churn" if proba >= THRESHOLD else "not_churned",
            "explanation": "OPENAI_API_KEY is not configured.",
        }

    prompt = build_prompt(str(user_id), proba, reasons)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=220,
    )

    explanation = response.choices[0].message.content.strip()

    return {
        "user_id": str(user_id),
        "churn_probability": proba,
        "predicted_label": "vol_churn" if proba >= THRESHOLD else "not_churned",
        "explanation": explanation,
    }

@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Churn Analyst</title>
  <style>
    :root {
      --bg: #07111f;
      --panel: rgba(255,255,255,0.06);
      --panel-2: rgba(255,255,255,0.08);
      --border: rgba(255,255,255,0.10);
      --text: #ecf3ff;
      --muted: #9fb1c9;
      --accent: #7c9cff;
      --danger: #ff7b8d;
      --warn: #ffcc7a;
      --ok: #87e0a3;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: radial-gradient(circle at top right, #16233c 0%, var(--bg) 45%);
      color: var(--text);
    }
    .wrap { max-width: 1380px; margin: 0 auto; padding: 28px; }
    .topbar {
      display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 22px;
    }
    .title { font-size: 32px; font-weight: 700; letter-spacing: -0.03em; }
    .subtitle { color: var(--muted); margin-top: 8px; max-width: 720px; line-height: 1.6; }
    .button {
      border: 1px solid var(--border); background: var(--panel); color: var(--text);
      padding: 12px 16px; border-radius: 16px; cursor: pointer;
    }
    .grid { display: grid; gap: 16px; }
    .kpis { grid-template-columns: repeat(4, minmax(0, 1fr)); margin-bottom: 16px; }
    .main { grid-template-columns: 1.15fr 0.85fr; }
    .bottom { grid-template-columns: 0.95fr 1.05fr; margin-top: 16px; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 20px;
      backdrop-filter: blur(12px);
      box-shadow: 0 12px 36px rgba(0,0,0,0.18);
    }
    .card h3 { margin: 0 0 8px; font-size: 18px; }
    .small { color: var(--muted); font-size: 13px; }
    .kpi-value { font-size: 34px; font-weight: 700; margin-top: 12px; letter-spacing: -0.04em; }
    .hero {
      background: linear-gradient(135deg, rgba(124,156,255,0.22), rgba(255,255,255,0.05));
    }
    .hero p { color: #d5e0f2; line-height: 1.7; }
    .pill {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 7px 11px; border-radius: 999px;
      background: rgba(124,156,255,0.16); color: #dce5ff; font-size: 12px;
      border: 1px solid rgba(124,156,255,0.24);
    }
    .feature-list, .user-list { display: flex; flex-direction: column; gap: 12px; }
    .row {
      display: flex; justify-content: space-between; gap: 12px; align-items: center;
      background: var(--panel-2); border: 1px solid var(--border); border-radius: 18px; padding: 12px 14px;
    }
    .bar { height: 8px; border-radius: 999px; background: rgba(255,255,255,0.08); overflow: hidden; margin-top: 8px; }
    .bar > span { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #8ba2ff, #bfd0ff); }
    .reason-positive { color: var(--danger); }
    .reason-negative { color: var(--ok); }
    .risk-high { color: var(--danger); }
    .risk-med { color: var(--warn); }
    .user-card-title { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
    .prob { font-size: 48px; font-weight: 800; letter-spacing: -0.05em; margin: 14px 0; }
    .muted-box {
      background: rgba(255,255,255,0.04); border: 1px solid var(--border); border-radius: 18px; padding: 14px;
    }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .loader { color: var(--muted); }
    @media (max-width: 1100px) {
      .kpis, .main, .bottom { grid-template-columns: 1fr; }
      .topbar { flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <div class="title">AI Churn Analyst</div>
        <div class="subtitle">Fast view of who will voluntarily cancel, why they are risky, and what signals matter most.</div>
      </div>
      <button class="button" onclick="loadDashboard()">Refresh data</button>
    </div>

    <div id="kpis" class="grid kpis"></div>

    <div class="grid main">
      <div class="card hero">
        <div class="pill">AI summary</div>
        <h3 style="margin-top:16px; font-size: 24px;">Why churn happens</h3>
        <p>
          This interface focuses on explainability. It combines churn probability, top behavioral drivers,
          and a human-readable summary so product teams can understand risk instead of only seeing a score.
        </p>
        <div class="two-col" style="margin-top:16px;">
          <div class="muted-box"><div class="small">Main question</div><div style="margin-top:8px; font-weight:600;">Will this user cancel?</div></div>
          <div class="muted-box"><div class="small">Main question</div><div style="margin-top:8px; font-weight:600;">Why are they likely to cancel?</div></div>
        </div>
      </div>
      <div class="card">
        <h3>Top global drivers</h3>
        <div class="small">Average absolute impact across users</div>
        <div id="features" class="feature-list" style="margin-top:16px;"></div>
      </div>
    </div>

    <div class="grid bottom">
      <div class="card">
        <h3>Highest-risk users</h3>
        <div class="small">Top users ranked by voluntary churn probability</div>
        <div id="users" class="user-list" style="margin-top:16px;"></div>
      </div>
      <div class="card">
        <div class="user-card-title">
          <div>
            <h3 id="detail-title">User details</h3>
            <div class="small">Click a user to inspect reasons</div>
          </div>
          <button class="button" id="explainBtn" onclick="generateExplanation()" disabled>Generate AI explanation</button>
        </div>
        <div id="detail" class="loader" style="margin-top:16px;">Select a user</div>
      </div>
    </div>
  </div>

  <script>
    let selectedUserId = null;

    function pct(v) {
      return `${(v * 100).toFixed(1)}%`;
    }

    function riskLabel(v) {
      if (v >= 0.7) return '<span class="risk-high">High</span>';
      if (v >= 0.4) return '<span class="risk-med">Medium</span>';
      return '<span class="reason-negative">Low</span>';
    }

    async function loadDashboard() {
      const [statsRes, usersRes] = await Promise.all([
        fetch('/stats'),
        fetch('/users?limit=12'),
      ]);

      const stats = await statsRes.json();
      const users = await usersRes.json();

      document.getElementById('kpis').innerHTML = `
        <div class="card"><div class="small">Users analyzed</div><div class="kpi-value">${stats.users}</div></div>
        <div class="card"><div class="small">Average risk</div><div class="kpi-value">${pct(stats.avg_risk)}</div></div>
        <div class="card"><div class="small">High-risk users</div><div class="kpi-value">${stats.high_risk_users}</div></div>
        <div class="card"><div class="small">API status</div><div class="kpi-value">Ready</div></div>
      `;

      document.getElementById('features').innerHTML = stats.top_features.map((item, i) => `
        <div>
          <div class="row">
            <div>
              <div style="font-weight:600;">${i + 1}. ${item.feature}</div>
              <div class="small">avg abs impact: ${Number(item.abs_shap).toFixed(3)}</div>
            </div>
          </div>
          <div class="bar"><span style="width:${Math.min(Number(item.abs_shap) * 100, 100)}%"></span></div>
        </div>
      `).join('');

      document.getElementById('users').innerHTML = users.map((user) => `
        <button class="row" style="text-align:left; cursor:pointer; color:inherit;" onclick="loadUser('${user.user_id}')">
          <div>
            <div style="font-weight:600;">${user.user_id}</div>
            <div class="small">Main positive driver: ${user.top_positive_driver ?? '—'}</div>
          </div>
          <div>
            <div style="font-weight:700;">${pct(user.churn_probability)}</div>
            <div class="small">${riskLabel(user.churn_probability)}</div>
          </div>
        </button>
      `).join('');

      if (users.length > 0) {
        await loadUser(users[0].user_id);
      }
    }

    async function loadUser(userId) {
      selectedUserId = userId;
      document.getElementById('explainBtn').disabled = false;
      const res = await fetch(`/users/${userId}`);
      const data = await res.json();

      document.getElementById('detail-title').textContent = `User ${data.user_id}`;
      document.getElementById('detail').innerHTML = `
        <div class="muted-box">
          <div class="small">Churn probability</div>
          <div class="prob">${pct(data.churn_probability)}</div>
          <div class="small">Predicted label: ${data.predicted_label}</div>
        </div>
        <div style="margin-top:16px;">
          <div class="small" style="margin-bottom:10px;">Top reasons</div>
          ${data.top_reasons.map((r) => `
            <div class="row" style="margin-bottom:10px; align-items:flex-start;">
              <div>
                <div style="font-weight:600;">${r.feature}</div>
                <div class="small">value: ${r.feature_value}</div>
              </div>
              <div style="text-align:right;">
                <div class="${r.shap_value > 0 ? 'reason-positive' : 'reason-negative'}">${r.direction}</div>
                <div class="small">impact ${Number(r.shap_value).toFixed(3)}</div>
              </div>
            </div>
          `).join('')}
        </div>
        <div id="ai-explanation" style="margin-top:16px;"></div>
      `;
    }

    async function generateExplanation() {
      if (!selectedUserId) return;
      const container = document.getElementById('ai-explanation');
      container.innerHTML = '<div class="muted-box">Generating explanation...</div>';
      const res = await fetch(`/users/${selectedUserId}/explanation`);
      const data = await res.json();
      container.innerHTML = `
        <div class="muted-box">
          <div class="small">AI explanation</div>
          <div style="margin-top:10px; line-height:1.7; color:#dce6f7;">${data.explanation}</div>
        </div>
      `;
    }

    loadDashboard();
  </script>
</body>
</html>
    """


# Run locally:
# uvicorn app:app --reload
# Required packages:
# pip install fastapi uvicorn pandas python-dotenv openai
# .env example:
# OPENAI_API_KEY=your_key_here
# CSV_PATH=/absolute/path/to/shap_top_reasons_per_user.csv
