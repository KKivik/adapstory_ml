import joblib
import os

class ML_classification_comments:
    def __init__(self):
        # Загрузка TF-IDF векторайзера
        self.tfidf_vectorizer = joblib.load('ML/classification_comments/tfidf_vectorizer.pkl')
        # Загрузка модели классификации
        self.classifier = joblib.load('ML/classification_comments/model_4_TF-IDFandLogReg.pkl')

    def predict(self, text):
        self.tfidf_features = self.tfidf_vectorizer.transform([text])
        # Предсказание с помощью модели
        self.prediction = self.classifier.predict(self.tfidf_features)
        self.confidence = self.classifier.predict_proba(self.tfidf_features).tolist()[0]
        return {
            "sentiment": self.prediction,  # Тональность
            "confidence": self.confidence[1] if self.prediction else self.confidence[1],
            "keywords": ["отличная", "погода", "прогулка"]
        }

if __name__ == '__main__':
    os.chdir("..")
    os.chdir("..")
    print(os.getcwd())
    model = ML_classification_comments()
    print(model.predict('Ужасный курс'))
    print(model.predict('Плохой курс'))
    print(model.predict('ахуенчик'))
    print(model.predict('Мне очень понравилось'))
    print(model.predict('хорошо'))