import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Pandas_handle:
    def loaddataframe(self, filename: str):
        load_dotenv()
        # ดึง Path จาก .env และจัดการเรื่องเครื่องหมาย / ให้ถูกต้อง
        folder_path = os.getenv("FILEPATHRAW", "") 
        filepath = os.path.join(folder_path, filename)
        df = pd.read_csv(filepath)
        return df

class PrepareLinearData:
    def __init__(self, df):
        # ใช้ .copy() เพื่อป้องกันไม่ให้ไปแก้ไฟล์ต้นฉบับโดยไม่ตั้งใจ
        self.df = df.copy() 
        self.le = LabelEncoder()

    def encode_allCol(self):
        # วนลูปแปลงเฉพาะคอลัมน์ที่เป็น Object (Text)
        for col in self.df.select_dtypes(include="object").columns:
            # ต้องมีวงเล็บที่ LabelEncoder() ถ้าประกาศใหม่ข้างบน
            self.df[col] = self.le.fit_transform(self.df[col].astype(str))
        return self.df # คืนค่า DataFrame ที่แปลงแล้วกลับไป