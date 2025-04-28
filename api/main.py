# --- main.py ---
import logging
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
import pandas as pd
from io import StringIO
from .services.sequence_analyzer import analyze_sequences
from .services.color_similarity import precompute_dot_products, get_color_similarity

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("la_matriz_api")

app = FastAPI(
    title="La Matriz API",
    version="1.0.0",
    description="API for color sequence analysis and color similarity calculations."
)

# Serve static files (for logo)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global storage (simulate in-memory caching)
DOT_PRODUCTS = None
MAX_DOT_PRODUCT = None
COLORS_DF = None
USAGE_STATS = {"analyze_sequences_calls": 0, "color_similarity_calls": 0}

@app.on_event("startup")
async def startup_event():
    global COLORS_DF, DOT_PRODUCTS, MAX_DOT_PRODUCT
    try:
        COLORS_DF = pd.read_csv("/api/data/la_matrice_plus.csv")
        DOT_PRODUCTS, MAX_DOT_PRODUCT = precompute_dot_products(COLORS_DF)
        logger.info("Startup: Color data loaded and dot products computed.")
    except Exception as e:
        logger.error(f"Error loading startup data: {e}")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@app.post("/analyze-sequences", summary="Analyze color sequences", description="Upload color, sequence, and semantic mapping CSVs to perform clustering and momentum analysis.")
async def analyze_sequences_endpoint(colors_file: UploadFile = File(...),
                                      sequences_file: UploadFile = File(...),
                                      semantic_file: UploadFile = File(...),
                                      k: int = Form(5)):
    global USAGE_STATS
    try:
        colors_df = pd.read_csv(StringIO((await colors_file.read()).decode('utf-8')))
        sequences_df = pd.read_csv(StringIO((await sequences_file.read()).decode('utf-8')))
        semantic_mapping_df = pd.read_csv(StringIO((await semantic_file.read()).decode('utf-8')))

        result = analyze_sequences(colors_df, sequences_df, semantic_mapping_df, k=k)
        USAGE_STATS["analyze_sequences_calls"] += 1
        logger.info(f"Analyze sequences called. Total calls: {USAGE_STATS['analyze_sequences_calls']}")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error in analyze-sequences: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/color-similarity", summary="Calculate color similarity", description="Provide two color names to compute their RGB dot product and similarity percentile.")
async def color_similarity_endpoint(color1: str = Form(...), color2: str = Form(...)):
    global DOT_PRODUCTS, MAX_DOT_PRODUCT, USAGE_STATS
    if DOT_PRODUCTS is None or MAX_DOT_PRODUCT is None:
        logger.error("Dot products not initialized.")
        return JSONResponse(content={"error": "Server not initialized properly."}, status_code=500)

    result = get_color_similarity(color1.strip().lower(), color2.strip().lower(), DOT_PRODUCTS, MAX_DOT_PRODUCT)
    USAGE_STATS["color_similarity_calls"] += 1
    logger.info(f"Color similarity called. Total calls: {USAGE_STATS['color_similarity_calls']}")
    return JSONResponse(content=result)

@app.get("/", summary="Health Check", description="Returns API running status.")
async def root():
    return {"message": "La Matriz API is running."}

@app.get("/usage-stats", summary="Usage Statistics", description="Returns the API usage statistics.")
async def get_usage_stats():
    return USAGE_STATS

# Customize OpenAPI Schema to add logo
@app.get("/openapi.json", include_in_schema=False)
async def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="La Matriz API",
        version="1.0.0",
        description="API for color sequence analysis and color similarity calculations.",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "/static/la_matriz_logo.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema
