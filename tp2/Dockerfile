FROM python:3.10.14-slim

WORKDIR /app

COPY ./requirements.txt ./requirements.txt
COPY ./regression.joblib ./regression.joblib
COPY ./model_app.py ./model_app.py

RUN pip install -r requirements.txt

EXPOSE 6969

CMD ["python3", "model_app.py"]
