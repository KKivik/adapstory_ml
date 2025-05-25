from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, BooleanField, SelectField
from wtforms.validators import DataRequired

class MLForm(FlaskForm):
    text = TextAreaField('Текст для анализа', validators=[DataRequired()])
    model_type = SelectField('Модель', choices=[('Тональность текста')])
    model_num = SelectField('Выберите тип модели', choices=[('Logistic Regression + TF-IDF'), ('Logistic Regression + RuElectro')])