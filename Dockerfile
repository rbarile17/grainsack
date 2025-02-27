FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["ls", "-la", "."]
