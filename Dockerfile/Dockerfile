FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "../src/train.py"]
