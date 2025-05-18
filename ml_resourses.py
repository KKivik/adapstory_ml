from flask_restful import reqparse, abort, Api, Resource
from flask import current_app, request
from itsdangerous import Serializer
from dotenv import load_dotenv
from ML.classification_comments.classification_comments import analyzer
import os

load_dotenv()


class Verify(Resource):
    def post(self):
        token = request.json.get('token')
        s = Serializer(os.getenv('SECRET_KEY'))
        try:
            data = s.loads(token)
        except:
            return {'valid': False}
        return {'valid': True, 'user_id': data['id']}


class ML_cl_LogReg_and_TF_IDF(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('api_key')
        parser.add_argument('text', required=True)
        args = parser.parse_args()

        token = args['api_key']
        print(token)
        s = Serializer(os.getenv('SECRET_KEY'))
        try:
            data = s.loads(token)
        except:
            abort(404, message=f"Ваш API_key неверен")


        text = args['text']
        return analyzer.predict(text)
