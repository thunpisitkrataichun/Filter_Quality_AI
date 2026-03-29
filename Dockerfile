FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api/main.py .
# COPY model.pkl .  (ถ้าจำเป็นจริง ๆ)

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


# FROM python:3.10 → ใช้ Python

# WORKDIR /app → โฟลเดอร์ทำงาน

# COPY . . → เอาโค้ดเข้า container

# RUN pip install → ลง library

# CMD → คำสั่งรัน API

# (docker build -t PROJECT1 .) ก็อปในวงเล็บเท่านั้นนะ ต้องพิมพ์ . แบบวรรคด้วย