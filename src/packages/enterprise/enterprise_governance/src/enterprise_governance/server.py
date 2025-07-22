"""Server entry point for enterprise_governance package."""

from fastapi import FastAPI

app = FastAPI(title="Enterprise Governance Service")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)