# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.predictive_lead_score import router as lead_router
from app.sentiment_analysis import router as sentiment_router
from app.order_forecasting.views import router as order_forcast_router
from app.smart_task.views import router as smart_task_router    
import uvicorn



app = FastAPI(title="AI Sales Intelligence", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(lead_router)
app.include_router(sentiment_router)
app.include_router(order_forcast_router)
app.include_router(smart_task_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)