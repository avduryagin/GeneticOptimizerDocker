from app.calclib.generator import Generator
from app.calclib.generator import ClRe
from app.calclib.tf_features import TFFeatures
from numpy.lib import recfunctions as rfn
import os
import pickle
import numpy as np
import app.calclib.tf_models as tf_models
#from keras.models import load_model
class SVRGenerator(Generator):
    def __init__(self,*args,classifier=None, regressor=None, col=None,path=None,
                 modelfolder='models',regmodel='svr.sav',clmodel='rfc.sav',
                 colfile='col.npy',scalers_folder='scalers',yscaler='yscaler.sav',
                 scaler='scaler.sav',threshold=0.5,**kwargs):
        super().__init__(classifier=classifier, regressor=regressor, col=col,path=path,modelfolder=modelfolder,regmodel=regmodel,clmodel=clmodel,colfile=colfile)
        self.scaler_path=os.path.join(self.path,scalers_folder)

        self.yscaler=pickle.load(open(os.path.join(self.scaler_path, yscaler), 'rb'))
        self.scaler = pickle.load(open(os.path.join(self.scaler_path, scaler), 'rb'))
        self.threshold=threshold



    def svr_regressor(self,x:np.ndarray):
        sx=self.scaler.transform(x)
        y=self.regressor.predict(sx)
        sy=self.yscaler.inverse_transform(y)
        return sy

    def get_next(self, x=ClRe(c=np.array([], dtype=float), r=np.array([], dtype=float),
                              t=np.array([], dtype=float), s=np.array([], dtype=float), shape=np.array([], dtype=int)),
                 top=np.array([], dtype=float)):
        # прогнозирование класссификационной задачи
        prob = self.classifier.predict_proba(x.c)
        pred_mask = np.where(prob[:, 1] > self.threshold)[0]
        # pred_mask = np.array(np.argmax(prob, axis=1), bool)
        # if pred_mask[pred_mask == True].shape[0] == 0:
        if pred_mask.shape[0] == 0:
            return None, pred_mask, prob
        # для  1 прогнозируется следующая точка y
        delta = self.svr_regressor(x.r[pred_mask]).reshape(-1)
        prev = x.r[pred_mask][:, -1]

        #sdel=delta*x.s[pred_mask]

        #print('delta>3', sdel[sdel>3].shape[0],'delta>4', sdel[sdel>4].shape[0],'delta<0', sdel[sdel<0].shape[0])
        #delta = np.abs(y - prev)
        y = prev + delta
        emask = y == prev
        y[emask] = top[pred_mask][emask]
        y_hat=y* x.s[pred_mask]
        x_hat = x.get_items(mask=pred_mask)
        #r_tilde=x.r[:,0]
        x_hat.r[:, 0]=x_hat.r[:,0]+1
        x_hat.r[:,1]=y
        r_tilde=x_hat.r
        x_tilde, t_tilde, shape_tilde = self.get_new(x=x_hat.c, tau=y_hat, t=x_hat.t, shape=x_hat.shape)
        return ClRe(c=x_tilde, r=r_tilde, t=t_tilde, shape=shape_tilde, s=x.s[pred_mask]), pred_mask, prob[:, 1]

class TFGenerator(Generator):
    def __init__(self,*args,classifier=None, regressor=None, col=None,path=None,
                 modelfolder='models',regmodel='tf_reg.h5',clmodel='tf_binary.h5',
                 colfile='col.npy',scalers_folder='scalers',reg_scaler='scaler_tf_reg.sav',
                 cl_scaler='scaler_tf_binary.sav',threshold=0.5,reg_shrinkage=1.,**kwargs):
        classifier=tf_models.tf_binary(model_file=clmodel,scaler_file=cl_scaler,
                                       model_folder=modelfolder,scaler_folder=scalers_folder)
        regressor=tf_models.tf_reg(model_file=regmodel,scaler_file=reg_scaler,
                                       model_folder=modelfolder,scaler_folder=scalers_folder,shrinkage=reg_shrinkage)
        super().__init__(classifier=classifier, regressor=regressor, col=col,path=path,modelfolder=modelfolder,regmodel=regmodel,clmodel=clmodel,colfile=colfile)

        self.threshold=threshold
        self.feat=None
        self.generated=dict()

    def predict(self,feat=TFFeatures(),stop=10,cutofftail=False):
        self.feat=feat
        predicted=super().predict(x=self.feat.ClRe,top=self.feat.top,stop=stop,cutofftail=cutofftail)
        return predicted


    def get_next(self, x=ClRe(c=np.array([], dtype=float), r=np.array([], dtype=float),
                              t=np.array([], dtype=float), s=np.array([], dtype=float), shape=np.array([], dtype=int)),
                 top=np.array([], dtype=float)):
        # прогнозирование класссификационной задачи
        prob = self.classifier.predict_proba(x.c)
        pred_mask = np.where(prob[:, 1] > self.threshold)[0]
        if pred_mask.shape[0] == 0:
            return None, pred_mask, prob
        # для  1 прогнозируется следующая точка y
        delta = self.regressor.predict(x.r[pred_mask]).reshape(-1)
        prev = x.r[pred_mask][:, -1]
        y = prev + delta
        emask = y == prev
        y[emask] = top[pred_mask][emask]
        y_hat=y* x.s[pred_mask]
        x_hat = x.get_items(mask=pred_mask)
        x_hat.r[:, 0]=x_hat.r[:,0]+1
        x_hat.r[:,1]=y
        r_tilde=x_hat.r
        #x_tilde, t_tilde, shape_tilde = self.get_new(x=x_hat.c, tau=y_hat, t=x_hat.t, shape=x_hat.shape)
        x_tilde, t_tilde, shape_tilde = self.get_new_(x_hat,y_hat)
        return ClRe(c=x_tilde, r=r_tilde, t=t_tilde, shape=shape_tilde, s=x.s[pred_mask]), pred_mask, prob[:, 1]

    def get_new_(self, f=ClRe(),tau=np.array([])):
        #
        # [0:'ads', 1:'ads05',2:'ads1', 3'ads2', 4'ads3',
        # 5'ivl0', 6'ivl1', 7'ivl2',8'ivl3', 9'ivl4', 10'ivl5',
        # 11'nivl0', 12'nivl1', 13'nivl2', 14'nivl3', 15'nivl4',15'nivl5',
        # 17'wmean', 18'amean', 19'percent', 20'tau', 21'interval', 22'water', 23'length']
        # t-предыстория
        # tau -новые значения
        cl=[]
        t=[]

        for k,i in enumerate(f.indices):
            a = self.feat.data[i]['a'][0]
            b = self.feat.data[i]['b'][0]
            x=self.feat.data[i]['x'][0]
            interval = self.feat.data[i]['interval'][0]
            index=self.feat.data[i]['index'][0]
            new_generated=np.zeros(shape=3)
            tau_=tau[k]
            new_generated[0]=tau_
            new_generated[1]=x

            try:
                generated_=self.generated[i]
                generated=np.vstack([new_generated,generated_])
            except KeyError:
                self.generated[i]=[]
                self.generated[i].append(new_generated)
                generated=new_generated.reshape(1,-1)
            newcl=self.feat.getnewcl(index=index,generated=generated,a=a,b=b,interval=interval)
            newt=np.append(f.t[i],tau_)
            cl.append(newcl)
            t.append(newt)
            self.generated[i]=generated
        f.shape=f.shape+1
        cl=np.vstack(cl)
        try:
            cl_=rfn.structured_to_unstructured(np.squeeze(cl,axis=1),dtype=np.float32)
            t_=np.array(t,dtype=np.float32)
        except ValueError:
            print()

        return cl_,t_ , f.shape
