from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles # NEW IMPORT
from app.api.routes import router

app = FastAPI(title="Smart Tile AI API")

# --- NEW: Serve your tile images so they can be seen in the browser ---
# This makes your images available at http://localhost:8000/catalog/filename.jpg
app.mount("/catalog", StaticFiles(directory="catalog_tiles"), name="catalog")

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)