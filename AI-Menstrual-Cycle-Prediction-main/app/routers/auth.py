import bcrypt
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from models import Subject
from schemas import RegisterRequest, RegisterResponse

router = APIRouter(prefix="/auth", tags=["auth"])

# 1. РЕГИСТРАЦИЯ (Создаем новый аккаунт)
@router.post("/register", response_model=RegisterResponse)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    # Хешируем ПИН (для безопасности, чтобы никто не узнал его из базы)
    pin_bytes = body.pin.encode("utf-8")
    pin_hash = bcrypt.hashpw(pin_bytes, bcrypt.gensalt()).decode("utf-8")

    # Создаем запись в базе
    subject = Subject(
        pin_hash=pin_hash,
        consent_given=body.consent,
        study_group=body.study_group,
    )
    db.add(subject)
    db.commit()
    db.refresh(subject)

    return RegisterResponse(
        subject_id=subject.subject_id, # Убедись, что в models.py это поле называется subject_id
        message="Account created anonymously"
    )

# 2. ВХОД (Авторизация по ПИН-коду)
@router.post("/login")
def login(pin: str, db: Session = Depends(get_db)):
    # Ищем всех пользователей
    users = db.query(Subject).all()
    
    # Проверяем ПИН каждого (так как они захешированы)
    for user in users:
        if bcrypt.checkpw(pin.encode("utf-8"), user.pin_hash.encode("utf-8")):
            return {
                "subject_id": user.subject_id, 
                "message": "Welcome back! С возвращением!"
            }
    
    # Если прошли всех и не нашли совпадения
    raise HTTPException(status_code=404, detail="Пользователь с таким PIN не найден")