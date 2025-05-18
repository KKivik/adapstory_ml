import joblib
import os
import pandas as pd
import nltk
import re
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords



class ML_classification_comments:
    def __init__(self):
        nltk.download('stopwords')
        # Загрузка TF-IDF векторайзера
        self.tfidf_vectorizer = joblib.load('ML/classification_comments/tfidf_vectorizer.pkl')
        # Загрузка модели классификации
        self.classifier = joblib.load('ML/classification_comments/model_4_TF-IDFandLogReg.pkl')

        self.morph = MorphAnalyzer()
        self.russian_stopwords = set(stopwords.words('russian')) - {'не'}  # Исключаем "не" из стоп-слов

    def predict(self, text):
        str_of_word = self.smart_lemmatize_and_remove_stopwords(text)
        self.tfidf_features = self.tfidf_vectorizer.transform([str_of_word])
        lst_of_word = str_of_word.split()
        # Предсказание с помощью модели
        self.prediction = self.classifier.predict(self.tfidf_features)
        self.confidence = self.classifier.predict_proba(self.tfidf_features).tolist()[0]
        weights = pd.DataFrame({'words': self.tfidf_vectorizer.get_feature_names_out(),
                                'weights': self.classifier.coef_.flatten()})

        # Фильтрация слов перед сортировкой
        weights_filtered = weights[weights['words'].isin(lst_of_word)]  # <--- ключевое изменение

        weights_min = weights_filtered.sort_values(by='weights').head(3)['words'].tolist()   # <--- сортировка отфильтрованных данных
        weights_max = weights_filtered.sort_values(by='weights', ascending=False).head(3)['words'].tolist()

        return {
            "sentiment": "Плохой" if self.prediction else "Хороший",  # Тональность
            "confidence": self.confidence[1] if self.prediction else self.confidence[0],
            "keywords": weights_min if self.prediction else weights_max
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


analyzer = ML_classification_comments()

if __name__ == '__main__':
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())
    model = ML_classification_comments()
    print(model.predict('Ужасный курс'))
    print(model.predict('Плохой курс'))
    print(model.predict('ахуенчик'))
    print(model.predict('Мне очень понравилось'))
    print(model.predict('Очень хорошо'))