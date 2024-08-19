from app.calclib import engineering as en, generator as gn
from app.calclib.tf_features import TFFeatures
from app.calclib.models import TFGenerator
import pandas as pd
import numpy as np

from numpy.lib import recfunctions as rfn

def predict(json,*args,get='dict',drift=0.,col=None,path=None,
                 modelfolder='models',regmodel='tf_reg.h5',clmodel='tf_binary.h5',
                 colfile='col.npy',scalers_folder='scalers',reg_scaler='scaler_tf_reg.sav',
                 cl_scaler='scaler_tf_binary.sav',threshold=0.5,epsilon=1/12.,reg_shrinkage=1.,**kwargs):

    dtype = {'id_simple_sector': np.int32, 'd': np.float32, 'l': np.float32, 's': np.float32,
              'date_input': np.datetime64, 'status': object, 'status_date_bezd': np.datetime64,
              'date_avar': np.datetime64, 'worl_avar_first': np.float32, 'locate_simple_sector': np.float32,
              'sw': np.float32, 'date_end_remont': np.datetime64, 'locate_simple_sector_1': np.float32,
              'l_remont': np.float32, 'date_rem_before_avar': np.datetime64, 'locate_remont_avar': np.float32,
              'l_remont_before_avar': np.float32}
    to_rename = dict(
        {"id_simple_sector": 'ID простого участка', "d": 'D', "l": 'L', "s": 'S', "date_input": 'Дата ввода',
         "status": 'Состояние',
         "status_date_bezd": 'Дата перевода в бездействие', "date_avar": 'Дата аварии',
         "worl_avar_first": 'Наработка до отказа',
         "locate_simple_sector": 'Адрес от начала участка', "sw": 'Обводненность',
         "date_end_remont": 'Дата окончания ремонта',
         "locate_simple_sector_1": 'Адрес от начала участка_1', "l_remont": 'Длина ремонтируемого участка',
         "date_rem_before_avar": 'Дата ремонта до аварии', "locate_remont_avar": 'Адрес ремонта до аварии',
         "l_remont_before_avar": 'Длина ремонта до аварии'})

    if type(json) is pd.core.frame.DataFrame:
        #для тестирования алгоритмов
        data = json
    else:
        #data = pd.read_json(json, orient='split', dtype=dtype)
        data=pd.DataFrame(json)
        for ty in dtype.keys():
            data[ty]=data[ty].astype(dtype[ty])
        data.rename(columns=to_rename, inplace=True, copy=False)
    data._is_copy = False
    #data=pd.read_json(json,orient='split',dtype=dtype)
    #data.rename(columns=to_rename,inplace=True)
    model=tf_predictor(clmodel=clmodel,regmodel=regmodel,
                       colfile=colfile,modelfolder=modelfolder,
                       scalers_folder=scalers_folder,cl_scaler=cl_scaler,
                       reg_scaler=reg_scaler,threshold=threshold,
                       col=col,path=path,reg_shrinkage=reg_shrinkage)
    if data.shape[0]>0:
        model.fit(data,mode='bw',ident='ID простого участка',restricts=True,drift=drift,epsilon=epsilon,regnorm=np.array([1],dtype=np.int32))
        model.predict()
        model.fill()
    if get=='dict':
        return model.diction
    elif get=='df':
        return model.results
    else:
        return model





class predictor:
    def __init__(self, *args, clmodel='rfc.sav',regmodel = 'rfreg.sav',colfile='col.npy',**kwargs):
        self.data = pd.DataFrame([])
        self.feat = en.features()
        #regmodel = 'rfreg_test.sav', clmodel = 'rfc_test.sav'
        self.gen = gn.Generator(clmodel=clmodel,regmodel=regmodel, colfile=colfile)
        # self.columns=["ID простого участка","Адрес от начала участка","Наработка до отказа","interval","predicted","time_series","probab"]
        self.columns = ['id_simple_sector', 'locate_simple_sector', 'worl_avar_first',
                        'interval', 'predicted', 'time_series', 'probab', 'lbound', 'rbound']
        self.results = pd.DataFrame([], columns=self.columns)
        self.diction = [{x:np.nan for x in self.columns}]
        self.probab=np.array([])
        self.time_series=np.array([])

    def fit(self, data, *args, **kwargs):
        self.data = data
        en.inscribing(self.data, *args, **kwargs)
        self.feat.fit(self.data, *args, **kwargs)

    def predict(self):
        self.predicted = self.gen.predict(x=self.feat.ClRe, top=self.feat.horizon)
        if self.gen.p.shape[0]>0:
            self.probab = np.cumsum(np.cumprod(self.gen.p.T, axis=1), axis=1)
            self.time_series = np.multiply(self.gen.r.T, self.feat.s.reshape(-1, 1))

    def fill(self,*args,**kwargs):
        if self.predicted.shape[0]==0:
            return
        for i in np.arange(self.feat.data.shape[0]):
            self.results.loc[i, 'time_series'] = self.time_series[i].tolist()
            self.results.loc[i, 'probab'] = self.probab[i].tolist()
        self.results.loc[:, 'predicted'] = self.predicted
        self.results.loc[:, 'interval'] = self.feat.data['interval'].reshape(-1).astype(np.int32)
        index = self.feat.data['index'].reshape(-1)
        self.results.loc[:, self.columns[0]] = self.data.loc[index, "ID простого участка"].values.astype(np.int32)
        self.results.loc[:, self.columns[1:3]] = self.data.loc[
            index, ["Адрес от начала участка", "Наработка до отказа"]].values
        delta = self.data.loc[index, 'a'].values
        self.results.loc[:, ['lbound', 'rbound']] = np.add(
            rfn.structured_to_unstructured(self.feat.data[['a', 'b']]).reshape(-1, 2), delta.reshape(-1, 1))
        self.diction=self.results.to_dict(orient='records')
        # self.json = self.results.to_json(orient='records')
class tf_predictor (predictor):
    def __init__(self, *args, col=None,path=None,
             modelfolder='models',regmodel='tf_reg.h5',clmodel='tf_binary.h5',
             colfile='col.npy',scalers_folder='scalers',reg_scaler='scaler_tf_reg.sav',
             cl_scaler='scaler_tf_binary.sav',threshold=0.5, reg_shrinkage=1.,**kwargs):
        self.data = pd.DataFrame([])
        self.feat = TFFeatures()
        self.gen=TFGenerator(clmodel=clmodel,regmodel=regmodel,cl_scaler=cl_scaler,
                             reg_scaler=reg_scaler,modelfolder=modelfolder,
                             scalers_folder=scalers_folder,colfile=colfile,
                             threshold=threshold,col=col,path=path,reg_shrinkage=reg_shrinkage)
        self.columns = ['id_simple_sector', 'locate_simple_sector', 'worl_avar_first',
                        'interval', 'predicted', 'time_series', 'probab', 'lbound', 'rbound']
        self.results = pd.DataFrame([], columns=self.columns)
        self.diction = [{x: np.nan for x in self.columns}]
        self.probab = np.array([])
        self.time_series = np.array([])
    def predict(self):
        self.predicted = self.gen.predict(feat=self.feat)
        if self.gen.p.shape[0]>0:
            self.probab = np.cumsum(np.cumprod(self.gen.p.T, axis=1), axis=1)
            self.time_series = np.multiply(self.gen.r.T, self.feat.s.reshape(-1, 1))





