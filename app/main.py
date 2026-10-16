import mimetypes
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.color_harmony import warm_catalog
from app.services.visual_search import CATALOG_FOLDER

for folder in ("catalog_tiles", "outputs", "static"):
    os.makedirs(folder, exist_ok=True)

# Most catalog tiles are WebP. Python 3.12's built-in table lacks it, so the slim Docker
# image served them as application/octet-stream (and Windows' registry as text/plain).
mimetypes.add_type("image/webp", ".webp")

@asynccontextmanager
async def lifespan(_app):
    # Colour profiles for the "Best Match" badges: cheap (no model), but reading every
    # catalog image on the first upload would delay its analysis.
    threading.Thread(target=warm_catalog, args=(CATALOG_FOLDER,), daemon=True).start()
    yield


app = FastAPI(title="Smart Tile AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Render-Info"],
)

app.include_router(router)

# Tile images, e.g. http://localhost:8000/catalog/marble.jpg
app.mount("/catalog", StaticFiles(directory="catalog_tiles"), name="catalog")
# Frontend assets (app.js, demo rooms)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
