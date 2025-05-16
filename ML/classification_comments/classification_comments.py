import joblib


class ML_classification_comments:
    def __init__(self):
        # Загрузка TF-IDF векторайзера
        self.tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        # Загрузка модели классификации
        self.classifier = joblib.load('model_4_TF-IDFandLogReg.pkl')

    def prefict(self, text):
        self.tfidf_features = self.tfidf_vectorizer.transform([text])
        # Предсказание с помощью модели
        self.prediction = self.classifier.predict(self.tfidf_features)
