from flask import request
from app import app

import numpy as np
from flask_restful import Api, Resource
#import json


api = Api(app)

class Calculation(Resource):
    def __init__(self, **kwargs):
        self.models=None

    def post(self,*args,**kwargs):
        x=request.json

        return None

api.add_resource(Calculation, "/calc/")
