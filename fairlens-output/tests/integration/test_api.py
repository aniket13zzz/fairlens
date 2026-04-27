"""
Integration tests for FastAPI endpoints.
Uses httpx.AsyncClient with ASGI transport — no real server needed.
"""
import io
import pytest
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def sample_csv_bytes():
    df = pd.DataFrame({
        "outcome": ["1"] * 20 + ["0"] * 20,
        "group":   ["A"] * 20 + ["B"] * 20,
        "age":     list(range(40)),
        "score":   [float(i) for i in range(40)],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def biased_csv_bytes():
    df = pd.DataFrame({
        "outcome": ["1"] * 20 + ["0"] * 20,
        "group":   ["A"] * 20 + ["B"] * 20,
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def app_client():
    import httpx
    from backend.main import app
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health(app_client):
    async with app_client as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_upload_valid_csv(app_client, sample_csv_bytes):
    async with app_client as client:
        r = await client.post(
            "/api/upload",
            files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
        )
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert data["rows"] == 40
    assert len(data["columns"]) == 4


@pytest.mark.asyncio
async def test_upload_non_csv_rejected(app_client):
    async with app_client as client:
        r = await client.post(
            "/api/upload",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
    assert r.status_code == 422
    assert r.json()["error"] == "INVALID_DATASET"


@pytest.mark.asyncio
async def test_upload_oversized_rejected(app_client):
    big = b"a,b\n" + b"1,2\n" * (11 * 1024 * 1024 // 4)  # >10 MB
    async with app_client as client:
        r = await client.post(
            "/api/upload",
            files={"file": ("big.csv", big, "text/csv")},
        )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_analyze_full_flow(app_client, biased_csv_bytes):
    async with app_client as client:
        up = await client.post(
            "/api/upload",
            files={"file": ("test.csv", biased_csv_bytes, "text/csv")},
        )
        sid = up.json()["session_id"]

        r = await client.post("/api/analyze", json={
            "session_id": sid,
            "target_column": "outcome",
            "sensitive_column": "group",
            "positive_outcome": "1",
        })
    assert r.status_code == 200
    data = r.json()
    assert data["severity"] == "SEVERE"
    assert data["metrics"]["disparate_impact"] == 0.0
    assert data["metrics"]["passes_80_rule"] is False
    assert "group_stats" in data
    assert "feature_importance" in data
    assert "ai_explanation" in data


@pytest.mark.asyncio
async def test_analyze_session_not_found(app_client):
    async with app_client as client:
        r = await client.post("/api/analyze", json={
            "session_id": "BADID",
            "target_column": "outcome",
            "sensitive_column": "group",
            "positive_outcome": "1",
        })
    assert r.status_code == 404
    assert r.json()["error"] == "SESSION_NOT_FOUND"


@pytest.mark.asyncio
async def test_analyze_bad_column_returns_422(app_client, biased_csv_bytes):
    async with app_client as client:
        up = await client.post(
            "/api/upload",
            files={"file": ("test.csv", biased_csv_bytes, "text/csv")},
        )
        sid = up.json()["session_id"]
        r = await client.post("/api/analyze", json={
            "session_id": sid,
            "target_column": "nonexistent_col",
            "sensitive_column": "group",
            "positive_outcome": "1",
        })
    assert r.status_code == 422
    assert r.json()["error"] == "COLUMN_NOT_FOUND"


@pytest.mark.asyncio
async def test_fix_requires_analysis_first(app_client, biased_csv_bytes):
    async with app_client as client:
        up = await client.post(
            "/api/upload",
            files={"file": ("test.csv", biased_csv_bytes, "text/csv")},
        )
        sid = up.json()["session_id"]
        r = await client.post("/api/fix", json={"session_id": sid})
    assert r.status_code == 400
    assert r.json()["error"] == "ANALYSIS_REQUIRED"


@pytest.mark.asyncio
async def test_full_pipeline_upload_analyze_fix(app_client, sample_csv_bytes):
    async with app_client as client:
        up = await client.post(
            "/api/upload",
            files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
        )
        sid = up.json()["session_id"]

        await client.post("/api/analyze", json={
            "session_id": sid,
            "target_column": "outcome",
            "sensitive_column": "group",
            "positive_outcome": "1",
        })

        fix = await client.post("/api/fix", json={"session_id": sid})

    assert fix.status_code == 200
    data = fix.json()
    assert "fixed_metrics" in data
    assert "improvement" in data
    assert data["improvement"]["eeoc_compliant"] in (True, False)
    # Accuracy delta must be a real number or None — never a random-looking float
    acc = data.get("accuracy_delta")
    assert acc is None or isinstance(acc, float)


@pytest.mark.asyncio
async def test_demo_endpoint(app_client):
    demo_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "sample_data", "hiring_dataset.csv"
    )
    if not os.path.exists(demo_path):
        pytest.skip("Demo dataset not present")
    async with app_client as client:
        r = await client.get("/api/demo")
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == "DEMO01"
    assert data["suggested"]["target"] == "hired"


@pytest.mark.asyncio
async def test_error_response_shape(app_client):
    """Errors must return {error, message} — never raw exception text."""
    async with app_client as client:
        r = await client.post("/api/analyze", json={
            "session_id": "GHOST",
            "target_column": "x",
            "sensitive_column": "y",
            "positive_outcome": "1",
        })
    body = r.json()
    assert "error" in body
    assert "message" in body
    # Must not contain Python traceback markers
    assert "Traceback" not in body.get("message", "")
    assert "Traceback" not in body.get("error", "")
