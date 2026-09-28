from fastapi import FastAPI

from app.api.clv_delivery_payments.clv_delivery_endpoints import router as clv_delivery_router

app = FastAPI(title="Customer Intelligence Platform - CLV, Delivery & Payments API")

app.include_router(clv_delivery_router)


@app.get("/")
def health():
    return {"status": "ok"}
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8001, reload=True)
