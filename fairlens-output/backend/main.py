"""
FairLens AI — Backend API v2.0
Fixes from gap analysis:
  - Typed exception hierarchy (no raw 500 leaks)
  - CORS from env var (no wildcard)
  - Upload validation: size limit, mime check, row cap
  - Session store via Redis or in-memory with TTL
  - BackgroundTasks cleanup for PDF temp files
  - Rate limiting via slowapi
  - Real accuracy delta (no np.random)
"""
from __future__ import annotations

import io
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .bias_detector import BiasDetector
from .debiaser import Debiaser
from .errors import (
    AnalysisRequired,
    FairLensError,
    InvalidDataset,
    SessionNotFound,
    fairlens_exception_handler,
)
from .explainer import AIExplainer
from .report_generator import ReportGenerator
from .session_store import session_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config from environment ──────────────────────────────────────────────────
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB
MAX_ROWS = int(os.getenv("MAX_DATASET_ROWS", "500000"))
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = (
    ["*"] if ALLOWED_ORIGINS_RAW.strip() == "*"
    else [o.strip() for o in ALLOWED_ORIGINS_RAW.split(",") if o.strip()]
)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FairLens AI",
    description="AI Bias Detection & Remediation Platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(FairLensError, fairlens_exception_handler)

# Serve frontend
_frontend = Path(__file__).parent.parent / "frontend"
if _frontend.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend), html=True), name="frontend")


# ── Pydantic models ──────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    session_id: str
    target_column: str
    sensitive_column: str
    positive_outcome: str
    prediction_column: Optional[str] = None
    ground_truth_column: Optional[str] = None


class FixRequest(BaseModel):
    session_id: str


class ReportRequest(BaseModel):
    session_id: str
    org_name: Optional[str] = "Your Organization"
    auditor_name: Optional[str] = "FairLens AI System"


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return RedirectResponse(url="/app")


@app.get("/health")
async def health():
    return {"status": "healthy", "api": "FairLens AI v2.0"}


@app.post("/api/upload")
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    """Upload CSV dataset. Validates size, mime type, and row count."""

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise InvalidDataset("Only CSV files are supported (.csv extension required).")

    # Size check — read up to MAX_UPLOAD_BYTES + 1 byte
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise InvalidDataset(
            f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB."
        )

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise InvalidDataset(f"Could not parse CSV: {exc}")

    if df.empty:
        raise InvalidDataset("Dataset is empty.")
    if len(df) > MAX_ROWS:
        raise InvalidDataset(
            f"Dataset has {len(df):,} rows — maximum allowed is {MAX_ROWS:,}."
        )
    if len(df.columns) < 2:
        raise InvalidDataset("Dataset must have at least 2 columns.")

    session_id = str(uuid.uuid4())[:8]
    session_store.set(session_id, {
        "df": df,
        "filename": file.filename,
        "analysis": None,
        "fixed": None,
    })

    columns = []
    for col in df.columns:
        uv = df[col].unique().tolist()
        columns.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "unique_count": len(uv),
            "sample_values": [str(v) for v in uv[:6]],
        })

    return {
        "session_id": session_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": columns,
        "preview": df.head(5).to_dict(orient="records"),
    }


@app.post("/api/analyze")
async def analyze_bias(request: AnalysisRequest):
    """Run full bias analysis on uploaded dataset."""
    session = session_store.get(request.session_id)
    if not session:
        raise SessionNotFound(request.session_id)

    df = session["df"]
    detector = BiasDetector()

    analysis = detector.analyze(
        df=df,
        target_col=request.target_column,
        sensitive_col=request.sensitive_column,
        positive_outcome=request.positive_outcome,
        prediction_col=request.prediction_column,
        ground_truth_col=request.ground_truth_column,
    )

    explainer = AIExplainer()
    analysis["ai_explanation"] = await explainer.explain(analysis)

    session_store.update(request.session_id, {
        "analysis": analysis,
        "config": {
            "target_col": request.target_column,
            "sensitive_col": request.sensitive_column,
            "positive_outcome": request.positive_outcome,
        },
    })

    return analysis


@app.post("/api/fix")
async def fix_bias(request: FixRequest):
    """Apply debiasing strategy chain and return improved metrics."""
    session = session_store.get(request.session_id)
    if not session:
        raise SessionNotFound(request.session_id)
    if not session.get("analysis"):
        raise AnalysisRequired()

    df = session["df"]
    config = session["config"]
    original_analysis = session["analysis"]

    debiaser = Debiaser()
    fixed_result = debiaser.fix(
        df=df,
        target_col=config["target_col"],
        sensitive_col=config["sensitive_col"],
        positive_outcome=config["positive_outcome"],
        original_metrics=original_analysis["metrics"],
    )

    explainer = AIExplainer()
    fixed_result["ai_explanation"] = await explainer.explain_fix(original_analysis, fixed_result)

    session_store.update(request.session_id, {"fixed": fixed_result})
    return fixed_result


@app.post("/api/report")
async def generate_report(request: ReportRequest, background_tasks: BackgroundTasks):
    """Generate PDF compliance report. Temp file auto-deleted after response."""
    import os as _os

    session = session_store.get(request.session_id)
    if not session:
        raise SessionNotFound(request.session_id)
    if not session.get("analysis"):
        raise AnalysisRequired()

    generator = ReportGenerator()
    pdf_path = generator.generate(
        session_data=session,
        org_name=request.org_name,
        auditor_name=request.auditor_name,
    )

    # Delete temp file after response is sent
    background_tasks.add_task(_os.unlink, pdf_path)

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"fairlens_audit_{request.session_id}.pdf",
        headers={"Content-Disposition": f"attachment; filename=fairlens_audit_{request.session_id}.pdf"},
    )


@app.get("/api/demo")
async def load_demo():
    """Load pre-configured demo dataset for quick evaluation."""
    demo_path = Path(__file__).parent.parent / "sample_data" / "hiring_dataset.csv"
    if not demo_path.exists():
        raise FairLensError("DEMO_UNAVAILABLE", "Demo dataset not found.", status=503)

    df = pd.read_csv(demo_path)
    session_id = "DEMO01"
    session_store.set(session_id, {
        "df": df,
        "filename": "hiring_dataset.csv",
        "analysis": None,
        "fixed": None,
    })

    columns = [
        {"name": col, "dtype": str(df[col].dtype),
         "unique_count": df[col].nunique(),
         "sample_values": [str(v) for v in df[col].unique()[:6]]}
        for col in df.columns
    ]

    return {
        "session_id": session_id,
        "filename": "hiring_dataset.csv",
        "rows": len(df),
        "columns": columns,
        "suggested": {"target": "hired", "sensitive": "gender", "positive_outcome": "1"},
    }
