from fastapi import FastAPI
from app.api import routes_upload

app = FastAPI(title="auto-ml")


# Health
@app.get("/health")
def health(): 
    return {
        "message" : "app is running"
    }

# Routes 
app.include_router(routes_upload.router)