import pandas as pd
import numpy as np
import os
import pickle
from app.calclib import models
from app.calclib import repair_programms as rp
from app.calclib import sets_methods as sm
from app.calclib import  tf_models


n=20
a=0
b=1000
a_=0
b_=5
z=np.empty(shape=(n,2))
age=np.random.random(n)*(b_-a_)+a_
x=np.random.random(n)*(b-a)+a
z[:,0]=x
z[:,1]=age

cover=sm.cover(z)
def price(x):
    if x<=168:
        return 16.156
    if (x>168)&(x<=219):
        return 18.939
    else:
        return 28.082


spath='C:\\Users\\avduryagin\\PycharmProjects\\tf_project'
fpath='D:\\Python\\ppd'
enscribed=pd.read_csv(os.path.join(fpath,'enscribed_schemes.csv'),parse_dates=['Дата ввода','Дата аварии',
                                                                               'Дата перевода в бездействие','Дата ремонта до аварии',
                                                                              'Дата окончания ремонта'])


mask=enscribed.loc[:,'scheme']=='nng'
xdata=enscribed.loc[mask]
grouped=xdata.groupby("ID простого участка")
columns=["ID простого участка","new_id","Дата аварии","Первичный отказ","Предотвращенный отказ","Пропущенный отказ","Ущерб первичнго отказа, млн.руб.","Длина сегмента, м","Число отказов","Ущерб-Прогноз, млн.руб.",
"Стоимость ремонтных меропритяий, мн.руб.","Прогнозная Эффективность", "Объемы Ремонта, м","Затраты на ремонт, млн.руб.",
        "Стоимость надежности, млн.руб.","Число предотвр.отказов","Число непредотвр. отказов","Ущерб предотвр. отказов, млн.руб.",
        "Объемы Ремонта,м","Объемы Ремонта,м (восстановленные)","Затраты на ремонт",'Затраты на ремонт (восстановленный)',"Демонтаж,м","Стоимость надежности","lbound","rbound","count","Адрес от начала участка","D", "L", "S"]
missed=[]
l=[]
cncall=0
rncall=0
keys=[18177]
for g in keys:

    index=grouped.groups[g]
    group=xdata.loc[index]
    #mask=np.ones(group.shape[0],dtype=bool)
    mask=group['Дата аварии']>=np.datetime64('2015-01-01')
    if mask[mask].shape[0]==0:
        missed.append(g)
        continue
    clmodel = 'ft_binary_do2.h5'
    #model_=models.TFGenerator(clmodel=clmodel)
    model_=models.SVRGenerator()
    cov=rp.pipe_cover(group,mask=mask,model=model_)
    cov.rep_price=price
    cov.fit()
    cl_call=cov.model.model.classifier.call_counter
    cncall+=cl_call
    r_call = cov.model.model.regressor.call_counter
    rncall+=r_call
    #print('rcalls {0}; total: {2}, ccalls {1}; total {3}'.format(r_call,cl_call,rncall,cncall))
    #rmask=np.isnan(group['lbound'])
    rmask=np.zeros(group.shape[0],dtype=bool)
    data=group.loc[~rmask,columns].values
    l.append(data)
    #print(g)
joined=np.vstack(l)
df=pd.DataFrame(joined,columns=columns)


#import app.calclib.pipeml as pml
#import app.calclib.remtime as remtime
#import app.calculation as calc

#import remtime as rm
#import json


import app.calclib.repair_programms as rp
path="C:\\Users\\avduryagin\\etc"
file='inscribed.csv'
dates=['Дата ввода','Дата аварии','Дата перевода в бездействие','Дата окончания ремонта','Дата ремонта до аварии']
xdata=pd.read_csv(os.path.join(path,file),parse_dates=dates,infer_datetime_format=True, dayfirst=True,engine='c')
group=xdata[xdata['ID простого участка']==62468]
mask=group['Дата аварии']>=np.datetime64('2015-01-01')

cov=rp.pipe_cover(group,mask=mask)
cov.fit()