"""
FastAPI routes.

POST /solve  — upload an image, get back a JSON solution.
GET  /health — liveness check.
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from backend.preprocessing.image_prep import preprocess_image
from backend.ocr.printed import extract_expression as printed_ocr
from backend.ocr.handwritten import extract_expression as handwritten_ocr
from backend.solver.solver import solve

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


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
        # "printed" or "auto" — use EasyOCR (best for printed; fallback for auto)
        raw_expr = printed_ocr(preprocessed)

    if not raw_expr.strip():
        raise HTTPException(
            status_code=422,
            detail="Could not extract any text from the image. "
                   "Try a clearer photo or switch OCR mode.",
        )

    result = solve(raw_expr)
    return JSONResponse(content=result)
