from fastapi import FastAPI
from app.api import routes_upload
from app.api import routes_training

app = FastAPI(title="auto-ml")


# Health
@app.get("/health")
def health():
    return {"message": "app is running"}


# Routes
app.include_router(routes_upload.router)
app.include_router(routes_training.router)
