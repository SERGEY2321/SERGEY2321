"""
FastAPI routes.

POST /solve     — upload an image, get back a JSON solution.
POST /generate  — generate a PNG image of a math expression.
GET  /random    — get a random expression string.
GET  /health    — liveness check.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse, Response

from backend.preprocessing.image_prep import preprocess_image
from backend.ocr.printed import extract_expression as printed_ocr
from backend.ocr.handwritten import extract_expression as handwritten_ocr
from backend.solver.solver import solve
from backend.utils.image_generator import (
    render_expression_image,
    random_expression,
    all_categories,
)

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/random")
def get_random():
    """Return a random math expression and its category."""
    return random_expression()


@router.get("/categories")
def get_categories():
    """Return all expression categories with example."""
    return all_categories()


@router.post("/generate")
async def generate_endpoint(
    expression: str = Form(...),
    width:  int = Form(520),
    height: int = Form(100),
):
    """
    Generate a PNG image of a math expression.

    Returns the PNG as binary (image/png).
    """
    expression = expression.strip()
    if not expression:
        raise HTTPException(status_code=400, detail="Expression cannot be empty")
    if len(expression) > 200:
        raise HTTPException(status_code=400, detail="Expression too long (max 200 chars)")

    png_bytes = render_expression_image(expression, width=width, height=height)
    return Response(content=png_bytes, media_type="image/png")


@router.post("/solve")
async def solve_endpoint(
    image: UploadFile = File(...),
    mode: str = Form("auto"),  # "auto" | "printed" | "handwritten"
):
    """
    Solve a math problem from an uploaded photo.

    - **image**: image file (JPEG, PNG, WEBP, …)
    - **mode**: OCR mode — `auto` (default), `printed`, or `handwritten`

    Returns JSON:
    ```json
    {
      "expression": "2*x + 3 = 7",
      "answer": "x = 2",
      "steps": ["…"],
      "verified": true,
      "error": null
    }
    ```
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        preprocessed = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Image preprocessing failed: {e}")

    if mode == "handwritten":
        raw_expr = handwritten_ocr(preprocessed)
    else:
        raw_expr = printed_ocr(preprocessed)

    if not raw_expr.strip():
        raise HTTPException(
            status_code=422,
            detail="Не удалось распознать текст на изображении. "
                   "Попробуйте более чёткое фото или смените режим OCR.",
        )

    result = solve(raw_expr)
    return JSONResponse(content=result)


@router.post("/solve-expression")
async def solve_expression_endpoint(expression: str = Form(...)):
    """
    Solve a math expression directly (without OCR step).
    Used when the user generates an image and solves it immediately.
    """
    expression = expression.strip()
    if not expression:
        raise HTTPException(status_code=400, detail="Expression cannot be empty")
    result = solve(expression)
    return JSONResponse(content=result)
