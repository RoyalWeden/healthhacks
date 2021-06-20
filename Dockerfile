FROM python:3.9.5-slim-buster

RUN mkdir /app

WORKDIR /app

ADD . .

RUN pip install -r requirements.txt

CMD gunicorn run:app --bind 0.0.0.0:$PORT --reload