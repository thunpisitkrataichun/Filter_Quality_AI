
# Krataichun Train Model

โปรเจคนี้จัดทำขึ้นเพื่อศึกษาการทำ Model ML, NN เพื่อนำไปต่อยอดสำหรับการทำโปรเจคในอนาคต นอกจากศึกษาการทำ Model เเล้ว หาก Model นั้นมีความน่าสนใจมากพอทางผู้จัดทำจะนำเอา Model ตัวนั้นไปทำ Restful api




## สารบัญ
- [ การติดตั้ง ]( #installation ) 
- [ การใช้งาน ]( #usage ) 
## Installation

To deploy this project run

```bash
  git clone https://github.com/thunpisitkrataichun/Filter_Quality_AI
  cd Filter_Quality_AI
```

สร้าง virtual environment
```bash
  python -m venv venv
```
Activate environment ถ้าทำถูกต้องจะมีวงเล็บขึ้นหน้า terminal เช่น (venv) C:\University\ 
```bash
venv\Scripts\activate
```
ให้ install library ใน ลง venv
```bash
pip install -r requirements.txt
```
Run the Local API server
```bash
uvicorn api.main:app --reload
```
ทดสอบ Api ผ่าน Browser

```bash
http://127.0.0.1:8000/docs
```


---

## 🌐 Live Demo

API is deployed on Render:

**Open API:**  
https://filter-quality-ai.onrender.com  

**Interactive API Docs:**  
https://filter-quality-ai.onrender.com/docs

## Usage
---
## API Reference


| Parameter | Type     |Required | Description                |
| :-------- | :------- |:--------|:------------------------- |
| `file` | `file` |  Yes |	Image file to upload|

Response (Example)
```bash
{
  "class": "cat",
  "confidence": 0.95
}
```

