import pandas as pd
import numpy as np
import os
import pickle
import json
from app.calclib import models
from app.calclib import repair_programms as rp
from app.calclib import sets_methods as sm
from app.calclib import  tf_models
import app.calclib.pipeml as pm
jspath="d:\\ml\\json"
with open(os.path.join(jspath,'1250002347.json'),'rb') as file:
    events=json.load(file)
data=events['crashes']['data']
args=events['crashes']['kwargs']
y=pm.predict(data,drift=args['drift'])
print(y)