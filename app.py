from flask import Flask, render_template, jsonify
from sqlalchemy.sql.functions import current_user

from data import db_session
from forms.user import RegisterForm
from forms.news import NewsForm
from forms.ml import MLForm
from data.users import User
from data.news import News
from flask import redirect, make_response, request, abort
from flask_login import LoginManager, login_user, logout_user, login_required
from flask_login import current_user
from forms.user import LoginForm
from ML.classification_comments.classification_comments import ML_classification_comments


app = Flask(__name__)
app.config['SECRET_KEY'] = 'yandexlyceum_secret_key'
analyzer = ML_classification_comments
login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    db_sess = db_session.create_session()
    return db_sess.query(User).get(user_id)

@app.route('/register', methods=['GET', 'POST'])
def reqister():
    form = RegisterForm()
    if form.validate_on_submit():
        if form.password.data != form.password_again.data:
            return render_template('register.html', title='Регистрация',
                                   form=form,
                                   message="Пароли не совпадают")
        db_sess = db_session.create_session()
        if db_sess.query(User).filter(User.email == form.email.data).first():
            return render_template('register.html', title='Регистрация',
                                   form=form,
                                   message="Такой пользователь уже есть")
        user = User(
            name=form.name.data,
            email=form.email.data,
        )
        user.set_password(form.password.data)
        db_sess.add(user)
        db_sess.commit()
        return redirect('/login')
    return render_template('register.html', title='Регистрация', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        db_sess = db_session.create_session()
        user = db_sess.query(User).filter(User.email == form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember_me.data)
            return redirect("/main")
        return render_template('login.html',
                               message="Неправильный логин или пароль",
                               form=form)
    return render_template('login.html', title='Авторизация', form=form)


@app.route('/', methods=['GET'])
def first_page():
    return render_template('first_page.html', title='Главная страница')

@app.route("/main")
def main_page():
    db_sess = db_session.create_session()
    news = db_sess.query(News).filter((News.is_private != True) | ((News.is_private == True) & (News.user_id == current_user.id)))
    # Создаем форму для ML
    ml_form = MLForm()
    result = None

    # Обрабатываем отправку формы
    if ml_form.validate_on_submit():
        text = ml_form.text.data
        model_type = ml_form.model_type.data
        # Вызываем ML-модель
        result = analyzer.predict(text)

    return render_template("index3.html", news=news, ml_form=ml_form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect("/")

@app.route('/news',  methods=['GET', 'POST'])
@login_required
def add_news():
    form = NewsForm()
    if form.validate_on_submit():
        db_sess = db_session.create_session()
        news = News()
        news.title = form.title.data
        news.content = form.content.data
        news.is_private = form.is_private.data
        current_user.news.append(news)
        db_sess.merge(current_user)
        db_sess.commit()
        return redirect('/main')
    return render_template('news.html', title='Добавление новости',
                           form=form)

@app.route('/news/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_news(id):
    form = NewsForm()
    if request.method == "GET":
        db_sess = db_session.create_session()
        news = db_sess.query(News).filter(News.id == id,
                                          News.user == current_user
                                          ).first()
        if news:
            form.title.data = news.title
            form.content.data = news.content
            form.is_private.data = news.is_private
        else:
            abort(404)
    if form.validate_on_submit():
        db_sess = db_session.create_session()
        news = db_sess.query(News).filter(News.id == id,
                                          News.user == current_user
                                          ).first()
        if news:
            news.title = form.title.data
            news.content = form.content.data
            news.is_private = form.is_private.data
            db_sess.commit()
            return redirect('/main')
        else:
            abort(404)
    return render_template('news.html',
                           title='Редактирование новости',
                           form=form
                           )



@app.route('/news_delete/<int:id>', methods=['GET', 'POST'])
@login_required
def news_delete(id):
    db_sess = db_session.create_session()
    news = db_sess.query(News).filter(News.id == id, News.user == current_user).first()
    if news:
        db_sess.delete(news)
        db_sess.commit()
    else:
        abort(404)
    return redirect('/main')



def main():
    db_session.global_init("db/blogs.db")
    app.run()


if __name__ == '__main__':
    main()