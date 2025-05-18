# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # Загружает переменные из .env

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'ML_AdapStory1242')