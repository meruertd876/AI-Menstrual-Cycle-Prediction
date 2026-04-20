from pydantic import BaseModel, Field, field_validator

class RegisterRequest(BaseModel):  # Проверь каждую букву!
    pin: str
    study_group: str | None = None
    consent: bool = True
    @field_validator("consent")
    @classmethod
    def must_consent(cls, v):
        if not v:
            raise ValueError("Согласие обязательно")
        return v

class RegisterResponse(BaseModel):
    subject_id: str
    message: str