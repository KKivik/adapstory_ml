from flask_restful import reqparse, abort, Api, Resource
from functools import wraps
from flask_login import current_user, login_required
from sqlalchemy.testing.plugin.plugin_base import config
from sqlalchemy_serializer import Serializer
from flask import current_app, request

from dotenv import load_dotenv
import os

load_dotenv()


class TokenResource(Resource):
    @login_required
    def get(self):
        return {'token': current_user.generate_auth_token()}

class Verify(Resource):
    def post(self):
        token = request.json.get('token')
        s = Serializer(os.getenv('SECRET_KEY'))
        try:
            data = s.loads(token)
        except:
            return {'valid': False}
        return {'valid': True, 'user_id': data['id']}