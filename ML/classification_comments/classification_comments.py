import joblib
import os
import pandas as pd
import nltk
import re
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from pathlib import Path

class ML_classification_comments:
    def __init__(self):
        nltk.download('stopwords')
        # Загрузка TF-IDF векторайзера
        base_dir = Path(__file__).parent.parent.parent  # Поднимаемся на два уровня вверх от текущего файла

        tfidf_path = base_dir / 'ML' / 'classification_comments' / 'LogRegandTF-IDF' / 'tfidf_vectorizer.pkl'
        model_path = base_dir / 'ML' / 'classification_comments' / 'LogRegandTF-IDF' / 'model_6_TF-IDFandLogReg.pkl'
        self.tfidf_vectorizer = joblib.load(tfidf_path)
        # Загрузка модели классификации
        self.classifier = joblib.load(model_path)

        self.morph = MorphAnalyzer()
        self.russian_stopwords = set(stopwords.words('russian')) - {'не'}  # Исключаем "не" из стоп-слов

        tokenyzer_path = base_dir / 'ML' / 'classification_comments' / 'tune_RuElectra'
        model_ruelectra_path = base_dir / 'ML' / 'classification_comments' / 'tune_RuElectra' / 'model.safetensors'

        self.tokenizer = AutoTokenizer.from_pretrained(tokenyzer_path)
        self.model_tune_ruelectra = AutoModelForSequenceClassification.from_pretrained(tokenyzer_path)
        self.classifier_ruelectra = pipeline(
            "text-classification",
            model=self.model_tune_ruelectra,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def predict_logreg_tfidf(self, text):
        str_of_word = self.smart_lemmatize_and_remove_stopwords(text)
        tfidf_features = self.tfidf_vectorizer.transform([str_of_word])

        lst_of_word = str_of_word.split()
        # Предсказание с помощью модели
        prediction = self.classifier.predict(tfidf_features)


        confidence = self.classifier.predict_proba(tfidf_features).tolist()[0]
        weights = pd.DataFrame({'words': self.tfidf_vectorizer.get_feature_names_out(),
                                'weights': self.classifier.coef_.flatten()})

        # Фильтрация слов перед сортировкой
        weights_filtered = weights[weights['words'].isin(lst_of_word)]  # <--- ключевое изменение

        weights_min = weights_filtered.sort_values(by='weights').head(3)['words'].tolist()   # <--- сортировка отфильтрованных данных
        weights_max = weights_filtered.sort_values(by='weights', ascending=False).head(3)['words'].tolist()

        return {
            "sentiment": "Плохой" if prediction else "Хороший",  # Тональность
            "confidence": confidence[1] if prediction else confidence[0],
            "keywords": weights_min if prediction else weights_max
        }

    def predict_tune_ruelectra(self, text):
        results = self.classifier_ruelectra(text)[0]
        return {
            "sentiment": "Плохой" if results['label'] == "LABEL_1" else "Хороший",  # Тональность
            "confidence": results['score'],
        }

    def smart_lemmatize_and_remove_stopwords(self, text):
        if not isinstance(text, str):  # Если не строка (на всякий случай)
            return str(text)
        words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text.lower())
        lemmas = []
        for word in words:
            if word not in self.russian_stopwords:
                lemmas.append(self.morph.parse(word)[0].normal_form)
            elif word == 'не':  # Сохраняем "не" как есть (без лемматизации)
                lemmas.append(word)
        return ' '.join(lemmas)


#analyzer = ML_classification_comments()

if __name__ == '__main__':
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())
    model = ML_classification_comments()
    print(model.predict_tune_ruelectra('Ужасный курс'))
