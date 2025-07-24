"""Data Management Server"""

from fastapi import FastAPI

app = FastAPI(title="Data Management Service")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "data-management"}

@app.get("/")
async def root():
    return {"message": "Data Management Service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)