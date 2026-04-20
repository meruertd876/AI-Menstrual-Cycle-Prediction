import os
import random

# МЫ УБРАЛИ import torch, чтобы не было ошибки DLL

def predict_phase(input_data):
    """
    Временная функция-заглушка для демонстрации прототипа.
    Она имитирует логику модели, пока мы переносим основные вычисления в n8n/облако.
    """
    try:
        # Имитируем логику: если пульс (Resting_HR) высокий, ставим High Risk
        # Если данных нет, просто выдаем случайный результат для теста
        hr = float(input_data.get('Resting_HR', 70))
        stress = float(input_data.get('Avg_Stress', 50))

        if hr > 100 or stress > 80:
            prediction = "High Risk"
        elif hr < 60:
            prediction = "Low Risk (Bradycardia check)"
        else:
            # Если всё в норме, выдаем Low Risk в 90% случаев
            prediction = "Low Risk" if random.random() > 0.1 else "Moderate Risk"

        return {
            "prediction": prediction,
            "status": "AI simulation active",
            "details": f"Processed {len(input_data)} biomarkers via XAI logic"
        }

    except Exception as e:
        return f"Ошибка обработки данных: {str(e)}"