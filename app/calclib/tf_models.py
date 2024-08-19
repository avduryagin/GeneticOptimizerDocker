import os
import pickle
import numpy as np
from keras.models import load_model
from tensorflow.nn import softmax
from tensorflow import convert_to_tensor
import tensorflow as tf
class tf_binary:
    def __init__(self, model_file='tf_binary.h5', scaler_file='scaler_tf_binary.sav', model_folder='models',
                 scaler_folder='scalers'):
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_folder)
        self.scaler_path = os.path.join(self.path, scaler_folder)
        self.scaler = pickle.load(open(os.path.join(self.scaler_path, scaler_file), 'rb'))
        self.model = load_model(os.path.join(self.path, model_file))
        self.call_counter=0
        self.input_shape=self.model.inputs[0].shape

    def predict_proba(self,x=np.array([]),softmax_=True):
        x_=self.scaler.transform(x)
        xhat=convert_to_tensor(x_,dtype=np.float32)
        y_=self.model(xhat).numpy()
        self.call_counter+=1
        y=y_
        if softmax_:
            y=softmax(y_).numpy()
        return y
    def predict(self,x=np.array([]),return_labels=True):
        y_=self.predict_proba(x)
        y=y_
        if return_labels:
            y=np.argmax(y_,axis=1)
        return y


class tf_reg:
    def __init__(self, model_file='tf_reg.h5', scaler_file='scaler_tf_reg.sav', model_folder='models',
                 scaler_folder='scalers',shrinkage=1.):
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_folder)
        self.scaler_path = os.path.join(self.path, scaler_folder)
        self.scaler = pickle.load(open(os.path.join(self.scaler_path, scaler_file), 'rb'))
        self.model = load_model(os.path.join(self.path, model_file))
        self.xindex=np.array([0,1],dtype=np.int32)
        self.yindex = np.array([2], dtype=np.int32)
        self.call_counter = 0
        self.input_shape = self.model.inputs[0].shape
        self.shrinkage = 1.
        if (shrinkage>=0.3)&(shrinkage<=1.):
            self.shrinkage=shrinkage


        # scaler x[:,[0,1]]  indices -x, scaler x[:,2]  indices -y
    def inverse_scale(self,x=np.array([]),index=np.array([0,1],dtype=np.int32)):
        data_min=self.scaler.data_min_[index]
        data_range=self.scaler.data_range_[index]
        x_=(x[:]*data_range)+data_min
        return x_
    def scale(self,x=np.array([]),index=np.array([0,1],dtype=np.int32)):
        data_min=self.scaler.data_min_[index]
        data_range=self.scaler.data_range_[index]
        x_=(x[:]-data_min)/data_range
        return x_



    def predict(self,x=np.array([]),apply_log=True,clip_to=0.125):
        xhat=x.copy()
        if apply_log:
            xhat[:,0]=np.log(xhat[:,0])+1
        x_=self.scale(xhat,self.xindex)
        xhat=convert_to_tensor(x_,dtype=np.float32)
        #print(self.model.predict(xhat, verbose=1))
        y_=self.model(xhat).numpy()
        #self.call_counter+=1
        y=self.inverse_scale(y_,self.yindex)
        mask=y[:,0]<clip_to
        y[mask,0]=clip_to
        return y.reshape(-1)*self.shrinkage









