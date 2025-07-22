"""Server entry point for enterprise_scalability package."""

from fastapi import FastAPI

app = FastAPI(title="Enterprise Scalability Service")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)