# FairLens AI v2.0 — Know Before You Deploy

> AI Bias Detection & Remediation Platform | EEOC · EU AI Act · NYC LL144

## Quick Start

```bash
# 1. Clone / unzip the project
cd fairlens-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Configure API keys for AI explanations
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY or OPENAI_API_KEY

# 4. Run
python run.py
```

Open **http://localhost:8000/app** in your browser.

---

## What It Does

FairLens AI helps organizations detect, quantify, and fix discriminatory bias in automated decision systems — hiring, lending, healthcare, education — before deployment.

### Fairness Metrics
| Metric | Description | Legal Threshold |
|--------|-------------|-----------------|
| **Disparate Impact Ratio** | Ratio of selection rates (min/max) | ≥ 0.80 (EEOC 80% Rule) |
| **Demographic Parity Diff** | Absolute selection rate gap | < 0.10 |
| **Equal Opportunity Diff** | True positive rate gap | < 0.10 |
| **Predictive Parity Diff** | Precision gap across groups | < 0.10 |
| **Proxy Feature Risk** | Pearson / Cramér's V correlation | — |

### Debiasing Strategies (Auto-Chain)
1. **Reweighing** (Preprocessing) — adjusts instance weights to equalize group selection
2. **Threshold Optimisation** (Post-Processing) — adjusts per-group decision thresholds

### Compliance Reports
PDF audit reports referencing EEOC, EU AI Act (Article 10), and NYC Local Law 144.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `GET`  | `/api/demo` | Load built-in hiring dataset |
| `POST` | `/api/upload` | Upload CSV dataset |
| `POST` | `/api/analyze` | Run bias analysis |
| `POST` | `/api/fix` | Apply auto-debiasing |
| `POST` | `/api/report` | Generate PDF compliance report |
| `GET`  | `/docs` | Interactive API documentation |

---

## Environment Variables

```env
# LLM (optional — falls back to rule-based explanations)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Session
SESSION_BACKEND=memory   # or redis
REDIS_URL=redis://localhost:6379

# Limits
MAX_UPLOAD_BYTES=10485760
MAX_DATASET_ROWS=500000

# Server
HOST=0.0.0.0
PORT=8000
```

---

## Architecture

```
fairlens-ai/
├── backend/
│   ├── main.py           # FastAPI app, routes
│   ├── bias_detector.py  # Fairness metrics engine
│   ├── debiaser.py       # Reweighing + threshold optimisation
│   ├── explainer.py      # AI explanation (Claude/GPT/fallback)
│   ├── report_generator.py  # PDF compliance report
│   ├── session_store.py  # In-memory / Redis session store
│   └── errors.py         # Typed exception hierarchy
├── frontend/
│   └── index.html        # Single-file React-free UI
├── sample_data/
│   └── hiring_dataset.csv
├── tests/
├── run.py
├── requirements.txt
└── .env.example
```

---

## Bug Fixes (v2.0)

- **Fixed**: `pandas.read_csv` AttributeError caused by environment package conflict
- **Fixed**: `cross_val_score fit_params` renamed to `params` in scikit-learn ≥ 1.4
- **Fixed**: API error messages (`data.detail` → `data.message`) not shown in frontend
- **Fixed**: numpy version constraint (`<2.3`) for scipy compatibility

## License
MIT
