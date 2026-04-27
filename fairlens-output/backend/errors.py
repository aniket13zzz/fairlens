"""
FairLens AI — Typed exception hierarchy.
No raw Python exceptions ever reach the client.
"""
from fastapi import Request
from fastapi.responses import JSONResponse


class FairLensError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        self.code = code
        self.message = message
        self.status = status
        super().__init__(message)


class SessionNotFound(FairLensError):
    def __init__(self, session_id: str):
        super().__init__(
            "SESSION_NOT_FOUND",
            f"Session '{session_id}' not found or expired. Re-upload your dataset.",
            status=404,
        )


class AnalysisRequired(FairLensError):
    def __init__(self):
        super().__init__(
            "ANALYSIS_REQUIRED",
            "Run /api/analyze before calling this endpoint.",
            status=400,
        )


class InvalidDataset(FairLensError):
    def __init__(self, reason: str):
        super().__init__("INVALID_DATASET", reason, status=422)


class ColumnNotFound(FairLensError):
    def __init__(self, col: str, available: list):
        super().__init__(
            "COLUMN_NOT_FOUND",
            f"Column '{col}' not found. Available columns: {available}",
            status=422,
        )


async def fairlens_exception_handler(request: Request, exc: FairLensError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content={"error": exc.code, "message": exc.message},
    )
