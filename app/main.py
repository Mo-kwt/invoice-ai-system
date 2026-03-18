from fastapi import FastAPI
from app.api.routes_upload import router as upload_router
from app.api.routes_invoice import router as invoice_router
from app.api.routes_dashboard import router as dashboard_router
from app.db.database import Base, engine
from app.db import models

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Invoice AI System")

app.include_router(upload_router, prefix="/api")
app.include_router(invoice_router, prefix="/api")
app.include_router(dashboard_router)


@app.get("/")
def read_root():
    return {"message": "System is working"}