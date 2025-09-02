FROM python:3.9

WORKDIR /spam_project

COPY ./requirements.txt /spam_project/requirements.txt

COPY ./model/model1.joblib /spam_project/model/model1.joblib

COPY ./model/vectorizer1.joblib /spam_project/model/vectorizer1.joblib

RUN pip install --no-cache-dir --upgrade -r /spam_project/requirements.txt

COPY ./app.py /spam_project/app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
