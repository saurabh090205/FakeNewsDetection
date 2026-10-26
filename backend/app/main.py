import logging
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import ALLOWED_ORIGINS

logger = logging.getLogger("veritas_ai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="Veritas AI — Fake News Detection & Explainability Platform",
    description="A deep learning and explainability platform evaluating news text patterns using neural architectures and LIME.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    # Enforce a 2MB maximum payload limit
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 2 * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={"error": "Payload too large. Maximum article size is 2MB.", "code": "PAYLOAD_TOO_LARGE"},
        )
    return await call_next(request)


app.include_router(router)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        payload = detail
    else:
        code = {
            400: "BAD_REQUEST",
            404: "NOT_FOUND",
            413: "PAYLOAD_TOO_LARGE",
            422: "VALIDATION_ERROR",
            500: "INTERNAL_SERVER_ERROR",
            503: "SERVICE_UNAVAILABLE",
        }.get(exc.status_code, "HTTP_ERROR")
        payload = {"error": str(detail), "code": code}
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    errors = exc.errors()
    first_msg = errors[0].get("msg", "Invalid input.") if errors else "Validation failed."
    return JSONResponse(
        status_code=422,
        content={"error": f"Validation error: {first_msg}", "code": "VALIDATION_ERROR"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.error("Unhandled server exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "An internal server error occurred during processing.", "code": "INTERNAL_SERVER_ERROR"},
    )

