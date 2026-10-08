import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

for folder in ("catalog_tiles", "outputs", "static"):
    os.makedirs(folder, exist_ok=True)

app = FastAPI(title="Smart Tile AI API")

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
