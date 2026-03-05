"""
DentalAI FastAPI Backend
Run with:  uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import traceback
from final import run_pipeline

app = FastAPI(title="DentalAI API", version="1.0")

# ── CORS: allow all origins (frontend served from file:// or localhost) ─────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


@app.get("/")
def root():
    return {"status": "DentalAI backend is running", "version": "1.0"}


@app.get("/health")
def health():
    """
    Health-check endpoint — called by the frontend before uploading.
    Returns 200 if the server is ready.
    """
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accept a dental OPG image, run the full AI pipeline, return JSON results.

    Expected response shape:
    {
      "teeth": [{"fdi": int, "condition": str, "confidence": float, "bbox": [x1,y1,x2,y2]}, ...],
      "missing_teeth": [int, ...],
      "missing_by_quadrant": {"upper_right": [...], "upper_left": [...], "lower_left": [...], "lower_right": [...]},
      "anomaly": {"label": str, "anomaly_score": float, "density_score": float, "height_score": float, "edge_score": float},
      "summary": {"total_detected": int, "total_missing": int, "implants": int, "cavities": int,
                  "impacted": int, "fillings": int, "normal": int}
    }
    """
    # Validate content type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Accepted: JPEG, PNG, WEBP."
        )

    # Save upload to disk temporarily
    safe_name = os.path.basename(file.filename or "upload.jpg")
    save_path = os.path.join(UPLOAD_FOLDER, safe_name)

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")

    # Run AI pipeline
    try:
        result = run_pipeline(save_path)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        # Always clean up the uploaded file
        try:
            os.remove(save_path)
        except Exception:
            pass

    # Check for pipeline-level errors returned as dict
    if result is None:
        raise HTTPException(status_code=500, detail="Pipeline returned no result.")
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result