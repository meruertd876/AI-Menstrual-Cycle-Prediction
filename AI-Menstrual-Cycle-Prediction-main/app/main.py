import sys
import os
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

# 1. Добавляем путь к текущей директории, чтобы Python видел соседние файлы
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# Добавляем также родительскую папку на всякий случай
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 2. Умный импорт
try:
    from database import engine, Base, get_db
    from models import Subject
    from ml_service import predict_phase
except ImportError:
    from app.database import engine, Base, get_db
    from app.models import Subject
    from app.ml_service import predict_phase

app = FastAPI()

# Создаем таблицы при старте
Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {
        "message": "PinkCycle API is running!",
        "status": "connected",
        "database": "online"
    }

@app.post("/predict")
def get_prediction(data: dict, db: Session = Depends(get_db)):
    """
    Браузер -> FastAPI -> Нейросеть -> База данных
    """
    try:
        # Здесь будет магия предсказания
        result = predict_phase(data)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))