FROM python:3.9

WORKDIR /app

COPY requirements.txt ./requirements.txt

COPY model1.joblib ./model1.joblib

COPY vectorizer1.joblib ./vectorizer1.joblib

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY app.py ./app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
