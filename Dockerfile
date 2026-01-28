FROM python:3.11-slim
WORKDIR /app

ENV PYSTOW_HOME=/lustrehome/robertobarile

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir grainsack

COPY ./grainsack ./grainsack
