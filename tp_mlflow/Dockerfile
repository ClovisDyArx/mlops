FROM python:3.10.14-slim

WORKDIR /app

COPY ./requirements.txt ./requirements.txt
COPY ./model.py ./model.py
COPY ./utils_mlflow.py ./utils_mlflow.py
COPY ./web_service.py ./web_service.py

RUN pip install -r requirements.txt

EXPOSE 1234

CMD ["python", "web_service.py"]