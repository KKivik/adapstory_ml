import os

from flask import Flask, redirect, render_template, url_for, request
from flask_wtf import FlaskForm
from loginform import LoginForm
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()
key_security = os.getenv('KEY')
app.config['SECRET_KEY'] = key_security
@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        return redirect('/success')
    return render_template('login.html', title='Авторизация', form=form)


@app.route('/index')
def ret():
    return 'И на Марсе будут яблони цвести!'


if __name__ == '__main__':
    app.run(port=8080, host='127.0.0.1')
