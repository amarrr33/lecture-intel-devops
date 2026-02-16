from fastapi import FastAPI

app = FastAPI(title="Low-Compute Multimodal Lecture Intelligence System (DevOps)")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "CI/CD running successfully"}
