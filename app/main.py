
from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

from app.risk_engine import (
    historical_var_cvar,
    gaussian_var_cvar,
    monte_carlo_var_cvar,
    multivariate_monte_carlo_var_cvar,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
# Local development uses the project directory.
# Render can set DB_PATH=/var/data/finrisk.sqlite3 on a persistent disk.
DB_PATH = Path(os.getenv("DB_PATH", str(ROOT / "finrisk.sqlite3")))

MODEL_VERSION = "baseline-1.2"

DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD"]

SUPPORTED_USD_SYMBOLS = {
    # Cryptocurrency
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "XRP-USD",

    # US equities
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",

    # US market indices
    "^GSPC",
    "^IXIC",

    # USD-priced commodities
    "CL=F",
    "BZ=F",
    "NG=F",
    "GC=F",
    "SI=F",
}
MONTE_CARLO_SIMULATIONS = 10_000
MONTE_CARLO_SEED = 42

app = FastAPI(
    title="FinRisk API",
    description=(
        "Quantitative portfolio risk research prototype. "
        "Risk estimates are uncertain and are not investment advice."
    ),
    version="1.1.0",
)


# Allow the frontend to call the API.
# Restrict FINRISK_FRONTEND_ORIGIN to your deployed frontend URL.
import os

frontend_origin = os.getenv("FINRISK_FRONTEND_ORIGIN", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------

class ForecastRequest(BaseModel):
    symbols: list[str] = Field(
        default_factory=lambda: DEFAULT_SYMBOLS.copy(),
        min_length=1,
        max_length=8,
    )
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "BTC-USD": 0.5,
            "ETH-USD": 0.5,
        }
    )
    portfolio_value: float = Field(default=1_000_000, gt=0)
    confidence: float = Field(default=0.95, gt=0.5, lt=0.999)
    period: str = Field(default="2y", pattern=r"^(1y|2y|5y)$")
    stress_shock: float = Field(default=-0.10, ge=-1.0, le=1.0)

    @model_validator(mode="after")
    def validate_portfolio(self):
        normalized_symbols = [
            symbol.strip().upper() for symbol in self.symbols
        ]

        if any(not symbol for symbol in normalized_symbols):
            raise ValueError("Symbols cannot be empty.")

        if len(set(normalized_symbols)) != len(normalized_symbols):
            raise ValueError("Symbols must be unique.")

        self.symbols = normalized_symbols

        unsupported = sorted(
            set(self.symbols) - SUPPORTED_USD_SYMBOLS
        )

        if unsupported:
            raise ValueError(
                "Unsupported instrument(s) for the USD-only MVP: "
                + ", ".join(unsupported)
            )

        normalized_weights = {
            str(symbol).strip().upper(): value
            for symbol, value in self.weights.items()
        }

        if set(normalized_weights) != set(self.symbols):
            raise ValueError(
                "Weights must contain exactly the selected symbols."
            )

        if any(
            not np.isfinite(weight) or weight < 0
            for weight in normalized_weights.values()
        ):
            raise ValueError(
                "Long-only weights must be finite and non-negative."
            )

        if not np.isclose(
            sum(normalized_weights.values()), 1.0, atol=1e-6
        ):
            raise ValueError("Portfolio weights must sum to 1.")

        self.weights = normalized_weights
        return self


# ---------------------------------------------------------------------
# SQLite forecast ledger and integrity verification
# ---------------------------------------------------------------------

def connect_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    with connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forecasts (
                forecast_id TEXT PRIMARY KEY,
                generated_at TEXT NOT NULL,
                data_cutoff TEXT NOT NULL,
                model_version TEXT NOT NULL,
                content_json TEXT NOT NULL,
                content_hash TEXT NOT NULL
            )
            """
        )


def record_without_hash(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key != "content_hash"
    }


def canonical_json(record: dict) -> str:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def save_forecast(record: dict) -> dict:
    canonical = canonical_json(record)
    digest = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO forecasts (
                forecast_id,
                generated_at,
                data_cutoff,
                model_version,
                content_json,
                content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                record["forecast_id"],
                record["generated_at"],
                record["data_cutoff"],
                record["model_version"],
                canonical,
                digest,
            ),
        )

    return {**record, "content_hash": digest}


def load_forecast(forecast_id: str) -> dict | None:
    with connect_db() as conn:
        row = conn.execute(
            "SELECT * FROM forecasts WHERE forecast_id = ?",
            (forecast_id,),
        ).fetchone()

    if row is None:
        return None

    record = json.loads(row["content_json"])
    stored_hash = row["content_hash"]

    actual_hash = hashlib.sha256(
        canonical_json(record_without_hash(record)).encode("utf-8")
    ).hexdigest()

    if actual_hash != stored_hash:
        raise RuntimeError(
            "Forecast integrity verification failed."
        )

    record["content_hash"] = stored_hash
    return record


initialize_db()



# ---------------------------------------------------------------------
# Market data and currency normalization
# ---------------------------------------------------------------------

INR_INSTRUMENTS = {
    "RELIANCE.NS",
    "TCS.NS",
    "^NSEI",
    "^BSESN",
}

FX_CONVERSION_SYMBOLS = {
    "INR": "USDINR=X",
}


def _download_close(symbol: str, period: str) -> pd.Series:
    """Download one instrument's adjusted closing prices."""
    try:
        frame = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise ValueError(
            f"Market-data request failed for {symbol}: {exc}"
        ) from exc

    if frame is None or frame.empty:
        raise ValueError(f"No market data returned for {symbol}.")

    if isinstance(frame.columns, pd.MultiIndex):
        if "Close" not in frame.columns.get_level_values(0):
            raise ValueError(
                f"No usable closing prices returned for {symbol}."
            )
        close = frame["Close"]
        if isinstance(close, pd.DataFrame):
            if symbol in close.columns:
                close = close[symbol]
            elif close.shape[1] == 1:
                close = close.iloc[:, 0]
            else:
                raise ValueError(
                    f"Ambiguous closing-price columns for {symbol}."
                )
    else:
        if "Close" not in frame.columns:
            raise ValueError(
                f"No usable closing prices returned for {symbol}."
            )
        close = frame["Close"]
        if isinstance(close, pd.DataFrame):
            if close.shape[1] != 1:
                raise ValueError(
                    f"Ambiguous closing-price columns for {symbol}."
                )
            close = close.iloc[:, 0]

    close = pd.to_numeric(close, errors="coerce")
    close = close.replace([np.inf, -np.inf], np.nan).dropna()
    close.name = symbol

    if close.empty or (close <= 0).any():
        raise ValueError(
            f"Invalid or non-positive prices returned for {symbol}."
        )

    return close


def fetch_prices(
    symbols: list[str],
    period: str,
) -> pd.DataFrame:
    """Return aligned USD-denominated prices for supported instruments."""
    if not symbols:
        raise ValueError("No instruments were selected.")

    raw_prices = {
        symbol: _download_close(symbol, period)
        for symbol in symbols
    }

    # USD/INR is quoted as INR per USD. Therefore INR prices are
    # divided by this rate to obtain their USD-equivalent prices.
    inr_symbols = [
        symbol for symbol in symbols
        if symbol in INR_INSTRUMENTS
    ]

    if inr_symbols:
        usd_inr = _download_close(
            FX_CONVERSION_SYMBOLS["INR"], period
        ).rename("USDINR=X")

        for symbol in inr_symbols:
            aligned_inr, aligned_fx = raw_prices[symbol].align(
                usd_inr, join="inner"
            )
            raw_prices[symbol] = (
                aligned_inr / aligned_fx
            ).rename(symbol)

    prices = pd.concat(
        [raw_prices[symbol] for symbol in symbols],
        axis=1,
        join="inner",
    ).sort_index()

    prices = prices.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(how="any")

    if len(prices) < 61:
        raise ValueError(
            "Fewer than 61 aligned price observations are available. "
            "Choose another period or supported instruments."
        )

    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Non-finite prices detected.")

    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive.")

    return prices



# ---------------------------------------------------------------------
# Quantitative risk engine
# ---------------------------------------------------------------------

def calculate_risk(
    prices: pd.DataFrame,
    request: ForecastRequest,
) -> dict:
    returns = prices.pct_change().dropna()

    weights = pd.Series(
        request.weights,
        dtype=float,
    ).reindex(returns.columns)

    if weights.isna().any():
        raise ValueError(
            "Portfolio weights do not match price columns."
        )

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("Portfolio weights must be finite.")

    portfolio_returns = returns @ weights

    if not np.isfinite(
        portfolio_returns.to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite portfolio returns detected.")

    # Historical and Gaussian estimates from the shared risk engine.
    historical = historical_var_cvar(
        portfolio_returns,
        confidence=request.confidence,
    )

    gaussian = gaussian_var_cvar(
        portfolio_returns,
        confidence=request.confidence,
    )

    # Univariate Gaussian Monte Carlo on portfolio returns.
    monte_carlo = monte_carlo_var_cvar(
        portfolio_returns,
        confidence=request.confidence,
        n_sims=MONTE_CARLO_SIMULATIONS,
        seed=MONTE_CARLO_SEED,
    )

    # Multivariate Gaussian Monte Carlo preserves asset covariance.
    multivariate_monte_carlo = None

    if len(returns.columns) >= 2:
        multivariate_monte_carlo = (
            multivariate_monte_carlo_var_cvar(
                returns,
                weights.to_dict(),
                confidence=request.confidence,
                n_sims=MONTE_CARLO_SIMULATIONS,
                seed=MONTE_CARLO_SEED,
            )
        )

    mu = gaussian["mean_return"]
    sigma = gaussian["volatility"]

    # Average weighted asset losses during the worst historical
    # portfolio-return observations.
    alpha = 1.0 - request.confidence

    cutoff = float(
        np.quantile(
            portfolio_returns,
            alpha,
            method="lower",
        )
    )

    tail_asset_returns = returns.loc[
        portfolio_returns <= cutoff
    ]

    contributions = {}

    if not tail_asset_returns.empty:
        for symbol in returns.columns:
            contributions[symbol] = float(
                -weights[symbol]
                * tail_asset_returns[symbol].mean()
            )

    stress = {
        symbol: {
            "weight": float(weights[symbol]),
            "shock": float(request.stress_shock),
            "pnl": float(
                request.portfolio_value
                * weights[symbol]
                * request.stress_shock
            ),
        }
        for symbol in weights.index
    }

    return {
        "observations": int(len(returns)),
        "data_cutoff": prices.index.max().isoformat(),
        "symbols": list(returns.columns),
        "weights": {
            key: float(value)
            for key, value in weights.items()
        },
        "risk": {
            "historical_var": historical["var"],
            "historical_cvar": historical["cvar"],
            "gaussian_var": gaussian["var"],
            "gaussian_cvar": gaussian["cvar"],
        },
        "monte_carlo": {
            "univariate": monte_carlo,
            "multivariate": multivariate_monte_carlo,
        },
        "volatility": sigma,
        "mean_daily_return": mu,
        "historical_tail_contributions": contributions,
        "stress": stress,
        "stress_total_pnl": float(
            sum(item["pnl"] for item in stress.values())
        ),
        "portfolio_value": float(request.portfolio_value),
        "confidence": float(request.confidence),
        "period": request.period,
        # Store losses to make the report chart reproducible.
        "portfolio_losses": (
            -portfolio_returns
        ).astype(float).tolist(),
    }


# ---------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
    }


@app.post("/api/forecasts")
def create_forecast(request: ForecastRequest):
    try:
        prices = fetch_prices(
            request.symbols,
            request.period,
        )

        results = calculate_risk(prices, request)

        record = {
            "forecast_id": uuid.uuid4().hex[:20],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_cutoff": results["data_cutoff"],
            "provider": "Yahoo Finance",
            "base_currency": "USD",
            "model_version": MODEL_VERSION,
            "symbols": results["symbols"],
            "weights": results["weights"],
            "portfolio_value": results["portfolio_value"],
            "confidence": results["confidence"],
            "period": results["period"],
            "observations": results["observations"],
            "risk": results["risk"],
            "monte_carlo": results["monte_carlo"],
            "volatility": results["volatility"],
            "mean_daily_return": results["mean_daily_return"],
            "historical_tail_contributions": (
                results["historical_tail_contributions"]
            ),
            "stress": results["stress"],
            "stress_total_pnl": results["stress_total_pnl"],
            "portfolio_losses": results["portfolio_losses"],
            "limitations": [
                "Historical, Gaussian, and Gaussian Monte Carlo "
                "baseline estimates.",
                "One-day horizon based on aligned daily observations.",
                "Multivariate Monte Carlo assumes a multivariate "
                "Gaussian return distribution.",
                "No EVT or deep-learning model is included.",
                "Market data may be delayed, revised, or incomplete.",
                "Risk estimates are not maximum-loss guarantees.",
                "Not investment advice.",
            ],
        }

        return save_forecast(record)

    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {exc}",
        ) from exc


@app.get("/api/forecasts/{forecast_id}")
def get_forecast(forecast_id: str):
    try:
        record = load_forecast(forecast_id)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Forecast not found.",
        )

    return record


@app.get("/api/forecasts/{forecast_id}/report.pdf")
def download_report(forecast_id: str):
    try:
        record = load_forecast(forecast_id)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Forecast not found.",
        )

    try:
        pdf = build_pdf_report(record)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PDF generation failed: {exc}",
        ) from exc

    filename = f"FinRisk_{forecast_id}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf),
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{filename}"'
            )
        },
    )


# ---------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------

def build_pdf_report(record: dict) -> bytes:
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="FinRisk Portfolio Risk Report",
        author="FinRisk",
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            spaceAfter=4,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontSize=12,
            leading=14,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=7,
            spaceAfter=5,
        )
    )

    story = []

    def section(title: str):
        story.append(
            Paragraph(title, styles["SectionTitle"])
        )

    def make_table(data, widths=None):
        table = Table(
            data,
            colWidths=widths,
            repeatRows=1,
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#17324D"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.35,
                        colors.HexColor("#D5DDE5"),
                    ),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [
                            colors.white,
                            colors.HexColor("#F2F5F8"),
                        ],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )

        return table

    risk = record["risk"]

    # Page 1: summary, portfolio, risk estimates and stress results.
    story.append(
        Paragraph(
            "FinRisk | Portfolio Risk Report",
            styles["Title"],
        )
    )

    story.append(
        Paragraph(
            "Quantitative risk analysis and decision support",
            styles["Normal"],
        )
    )

    story.append(Spacer(1, 6))

    metadata = [
        ["Forecast ID", record["forecast_id"]],
        ["Generated UTC", record["generated_at"]],
        ["Data cutoff", record["data_cutoff"]],
        ["Model version", record["model_version"]],
        [
            "Provider / base currency",
            f'{record["provider"]} / {record["base_currency"]}',
        ],
        [
            "Confidence / sample",
            (
                f'{record["confidence"]:.1%} / '
                f'{record["observations"]} returns'
            ),
        ],
        [
            "Portfolio value",
            (
                f'{record["base_currency"]} '
                f'{record["portfolio_value"]:,.2f}'
            ),
        ],
    ]

    story.append(
        make_table(
            [["Forecast metadata", "Value"]] + metadata,
            [48 * mm, 125 * mm],
        )
    )

    section("Portfolio composition")

    composition = [["Instrument", "Weight"]]
    composition += [
        [symbol, f"{weight:.2%}"]
        for symbol, weight in record["weights"].items()
    ]

    story.append(
        make_table(
            composition,
            [90 * mm, 83 * mm],
        )
    )

    section("Risk estimates")

    risk_rows = [
        ["Method", "VaR", "CVaR"],
        [
            "Historical simulation",
            f'{risk["historical_var"]:.2%}',
            f'{risk["historical_cvar"]:.2%}',
        ],
        [
            "Gaussian parametric",
            f'{risk["gaussian_var"]:.2%}',
            f'{risk["gaussian_cvar"]:.2%}',
        ],
    ]

    monte_carlo = record.get("monte_carlo", {})
    univariate = monte_carlo.get("univariate")
    multivariate = monte_carlo.get("multivariate")

    if univariate:
        risk_rows.append(
            [
                "Univariate Monte Carlo",
                f'{univariate["var"]:.2%}',
                f'{univariate["cvar"]:.2%}',
            ]
        )

    if multivariate:
        risk_rows.append(
            [
                "Multivariate Monte Carlo",
                f'{multivariate["var"]:.2%}',
                f'{multivariate["cvar"]:.2%}',
            ]
        )

    story.append(
        make_table(
            risk_rows,
            [77 * mm, 48 * mm, 48 * mm],
        )
    )

    story.append(
        Paragraph(
            "VaR is a loss threshold at the selected confidence level. "
            "CVaR estimates average loss in the modeled tail beyond that "
            "threshold. These are model-based estimates, not maximum-loss "
            "guarantees.",
            styles["SmallText"],
        )
    )

    section("Hypothetical stress scenario")

    stress_rows = [
        ["Instrument", "Weight", "Shock", "Estimated P&L"]
    ]

    for symbol, item in record["stress"].items():
        stress_rows.append(
            [
                symbol,
                f'{item["weight"]:.2%}',
                f'{item["shock"]:.1%}',
                f'{item["pnl"]:,.2f}',
            ]
        )

    stress_rows.append(
        [
            "Portfolio",
            "100%",
            "Scenario",
            f'{record["stress_total_pnl"]:,.2f}',
        ]
    )

    story.append(
        make_table(
            stress_rows,
            [42 * mm, 35 * mm, 35 * mm, 61 * mm],
        )
    )

    story.append(Spacer(1, 5))

    story.append(
        Paragraph(
            "Research prototype only. Results depend on data quality, "
            "portfolio assumptions, model specification, and market regime. "
            "This report is not investment advice.",
            styles["SmallText"],
        )
    )

    # Page 2: visualizations, contributions, limitations and hash.
    story.append(PageBreak())

    story.append(
        Paragraph(
            "Risk Visualizations & Interpretation",
            styles["Title"],
        )
    )

    labels = [
        "Historical VaR",
        "Gaussian VaR",
        "Historical CVaR",
        "Gaussian CVaR",
    ]

    values = [
        risk["historical_var"],
        risk["gaussian_var"],
        risk["historical_cvar"],
        risk["gaussian_cvar"],
    ]

    if univariate:
        labels += ["MC VaR", "MC CVaR"]
        values += [
            univariate["var"],
            univariate["cvar"],
        ]

    fig, ax = plt.subplots(figsize=(7.2, 2.6))

    bars = ax.bar(
        labels,
        np.array(values) * 100,
    )

    ax.set_ylabel("Estimated loss (%)")
    ax.set_title("VaR and CVaR comparison")
    ax.tick_params(
        axis="x",
        labelrotation=12,
        labelsize=8,
    )
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()

    chart1 = io.BytesIO()
    fig.savefig(
        chart1,
        format="png",
        dpi=140,
        bbox_inches="tight",
    )
    plt.close(fig)
    chart1.seek(0)

    story.append(
        Image(
            chart1,
            width=172 * mm,
            height=63 * mm,
        )
    )

    # Use the stored loss series. Do not download new market data.
    losses = np.asarray(
        record.get("portfolio_losses", []),
        dtype=float,
    )

    if len(losses) and np.isfinite(losses).all():
        fig, ax = plt.subplots(figsize=(7.2, 2.5))

        ax.hist(
            losses * 100,
            bins=35,
            density=True,
            alpha=0.75,
        )

        ax.axvline(
            risk["historical_var"] * 100,
            linestyle="--",
            label="Historical VaR",
        )

        ax.axvline(
            risk["historical_cvar"] * 100,
            linestyle=":",
            label="Historical CVaR",
        )

        ax.set_xlabel("Daily portfolio loss (%)")
        ax.set_ylabel("Density")
        ax.set_title("Stored portfolio loss distribution")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()

        chart2 = io.BytesIO()
        fig.savefig(
            chart2,
            format="png",
            dpi=140,
            bbox_inches="tight",
        )
        plt.close(fig)
        chart2.seek(0)

        story.append(
            Image(
                chart2,
                width=172 * mm,
                height=59 * mm,
            )
        )

    else:
        story.append(
            Paragraph(
                "Loss-distribution chart unavailable for this older "
                "forecast because its loss observations were not stored.",
                styles["SmallText"],
            )
        )

    section("Historical tail-risk contributions")

    contributions = record["historical_tail_contributions"]

    contribution_rows = [
        ["Instrument", "Estimated contribution"]
    ]

    contribution_rows += [
        [symbol, f"{value:.4%}"]
        for symbol, value in contributions.items()
    ]

    story.append(
        make_table(
            contribution_rows,
            [90 * mm, 83 * mm],
        )
    )

    section("Methodology and limitations")

    for limitation in record["limitations"]:
        story.append(
            Paragraph(
                f"â€¢ {limitation}",
                styles["SmallText"],
            )
        )

    content_hash = record.get(
        "content_hash",
        "Unavailable",
    )

    story.append(Spacer(1, 4))

    story.append(
        Paragraph(
            f"Verified forecast content hash: {content_hash}",
            styles["SmallText"],
        )
    )

    doc.build(story)
    return buffer.getvalue()

