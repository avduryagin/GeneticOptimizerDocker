from app.calclib.models import SVRGenerator
from app.calclib.features import SVRsingle,cover
import app.calclib.engineering as en
import pandas as pd
import numpy as np

class sbs:
    def __init__(self,data,mask=None,model=None,engine=None):
        self.columns=['Наработка до отказа','Адрес от начала участка','L',
         'Наработка до отказа(new), лет','Адрес от начала участка (new)','L,м','a']
        self.data=data
        self.model=model
        self.engine=engine
        if self.model is None:
            self.model=SVRGenerator()
        if self.engine is None:
            self.engine=SVRsingle()
        self.data._is_copy = False


        if mask is None:
            self.mask=np.ones(self.data.shape[0],dtype=bool)
        else:
            self.mask = mask

        self.cover = cover(self.data.loc[self.mask,self.columns].values)
        self.masked_index = self.data.index[self.mask]

        self.grouped=self.data.groupby('new_id').groups
        self.current_index=data.index[0]
        self.decision_func=lambda x: True if x>0.5 else False
        self.size_func = lambda x: 100 if x > 219 else 150
        self.regnorm=np.array([1],dtype=np.int32)


    def predict(self,expand_result=False):
        diameter = self.data.loc[self.current_index, 'D']
        size = self.size_func(diameter)
        self.cover.size=size
        res=self.cover.value()
        if res is None:
            return None
        i=res[0]
        a=res[1]
        b=res[2]
        self.current_index=self.masked_index[i]
        pipe=self.data.loc[self.current_index,'new_id']
        drift = self.data.loc[self.current_index, 'a']


        ahat=a-drift
        bhat=b-drift
        indices=self.grouped[pipe]
        self.engine.fit(self.data.loc[indices],index=self.current_index,lbound=ahat,rbound=bhat,ints=np.array(size),regnorm=self.regnorm,groups=self.grouped)
        predicted = self.model.predict(self.engine, cutofftail=False)

        if expand_result:
            target=self.engine.data['target'].reshape(-1)[0]
            count = self.engine.data['count'].reshape(-1)[0]
            n_predicted = np.where(self.model.p > 0)[0].shape[0]
            decision=self.decision_func(predicted[0])
            pred=predicted[0]
            return np.array([decision, a, b,count,target,pred,n_predicted],dtype=np.float32)
        else:
            return np.array([self.decision_func(predicted[0]),a,b],dtype=np.float32)

    def get_next(self,state=True):
            self.cover.get_index(state=state)

class pipe_cover:
    def __init__(self,data,mask=None,model=None,engine=None):
        self.data=data
        self.model=sbs(data,mask,model=model,engine=engine)
        self.columns=['lbound','rbound','length','predicted','n_predicted','count','target',
                      'lost_mask','predicted_mask','single_mask','repairs_length','synthetic_length']
        self.rep_price = lambda x: 18 if x > 219 else 12
        self.lost = lambda x: 3.
        self.cut_rep=np.datetime64('2015-01-01')

    def fit(self):
        i=0
        self.data.loc[self.data.index,self.columns]=np.nan
        while i<self.data.shape[0]:
            res = self.model.predict(expand_result=True)
            #print(res)

            if res is not None:
                state = bool(res[0])
                a = res[1]
                b = res[2]
                count=res[3]
                target=res[4]
                predicted=res[5]
                n_predicted=res[6]


                index = self.model.current_index
                self.data.at[index, 'lbound']=a
                self.data.at[index, 'rbound'] = b
                self.data.at[index, 'length'] = b-a
                self.data.at[index, 'predicted'] = predicted
                self.data.at[index, 'n_predicted'] = n_predicted
                self.data.at[index, 'count'] = count
                self.data.at[index, 'target'] = target
                self.model.get_next(state=state)
                #print(a,b)
            else:
                break
                # plt.scatter(group[~sub.mask]['Адрес от начала участка'],group[~sub.mask]['Наработка до отказа'],c='green')

            i += 1
        lost_mask=self.model.cover.marked&(~self.model.cover.predicted)
        predicted_mask=self.model.cover.predicted
        single_mask=(~predicted_mask)&(~lost_mask)
        indices=self.model.masked_index[self.model.cover.sa]
        #print(np.where(predicted_mask)[0].shape[0])

        self.data.loc[:,['lost_mask','predicted_mask','single_mask']]=False

        self.data.loc[indices,'lost_mask']=lost_mask
        self.data.loc[indices,'predicted_mask']=predicted_mask
        self.data.loc[indices,'single_mask'] = single_mask
        self.data.loc[:,'single_mask']=self.data.loc[:,'single_mask'].astype(np.float32)

        #date=self.data.loc[self.model.mask,'Дата аварии'].min()


        raw=en.get_raw_repairs(self.data[self.model.mask])
        rmask=raw['Дата ремонта']>=self.cut_rep
        raw=raw[rmask]
        unique=en.get_unical_repairs(self.data[self.model.mask])
        umask=unique['Дата ремонта']>=self.cut_rep
        unique=unique[umask]
        synthetic=en.get_merged_repairs(raw,unique)
        raw_length=0
        synthetic_length=0
        if raw.shape[0]>0:
            raw_length=raw['Длина'].sum()
            synthetic_length = synthetic['Длина'].sum()
        self.data.at[self.model.masked_index[0],'repairs_length'] = raw_length
        self.data.at[self.model.masked_index[0],'synthetic_length'] = synthetic_length

        status=set(self.data['Состояние'].value_counts().keys())
        if "Действующий" in status:
            ldm=0
        else:
            ldm=self.data.at[self.model.masked_index[0], 'L']

        self.data.at[self.model.masked_index[0], 'Демонтаж,м'] = ldm
        #self.data.at[self.model.masked_index[0], 'Число первичных отказов'] = np.where(self.data['single_mask'].astype(bool))[0].shape[0]
        mask=self.data.loc[self.model.masked_index,'predicted']==0
        zeros=self.model.masked_index[mask]

        lost=self.data['lost_mask'].values
        self.data.loc[:, 'Число непредотвр. отказов']=np.nan

        for z in zeros:

            a=self.data.at[z,'lbound']
            b=self.data.at[z,'rbound']
            t=self.data.at[z,'Наработка до отказа']
            lmask=(self.data['Адрес от начала участка']>=a)&(self.data['Адрес от начала участка']<=b)
            tmask=(self.data['Наработка до отказа'].values>=t)
            i=self.data.index.get_loc(z)
            tmask[i]=False
            rmask=lmask&tmask&lost
            lcount=np.where(rmask)[0]
            self.data.at[z,'Число непредотвр. отказов']=lcount.shape[0]

        self.data.loc[:, 'Ущерб первичнго отказа, млн.руб.'] = self.data.loc[:].apply(lambda x:self.lost(x['D']), axis=1)
        self.data.loc[:, 'Стоимость ремонтных меропритяий, мн.руб.'] = self.data.loc[:].apply(
            lambda x: self.rep_price(x['D']) * x['length'] / 1000., axis=1)
        self.data.loc[:, 'Затраты на ремонт, млн.руб.'] = self.data.loc[:].apply(
            lambda x: x['Стоимость ремонтных меропритяий, мн.руб.'] * self.model.decision_func(x['predicted']), axis=1)
        self.data.loc[:, 'Объемы Ремонта, м'] = self.data.loc[:].apply(
            lambda x: x['length'] * self.model.decision_func(x['predicted']), axis=1)
        self.data.loc[:, 'Ущерб-Прогноз, млн.руб.'] = self.data.loc[:].apply(
            lambda x: x['n_predicted'] * x['Ущерб первичнго отказа, млн.руб.'], axis=1)
        self.data.loc[:, 'Прогнозная Эффективность'] = self.data.loc[:].apply(
            lambda x: x['Ущерб-Прогноз, млн.руб.'] / x['Стоимость ремонтных меропритяий, мн.руб.'], axis=1)
        self.data.loc[:, 'Число предотвр.отказов'] = self.data.loc[:].apply(
            lambda x: x['count'] * self.model.decision_func(x['predicted']), axis=1)
        self.data.loc[:, 'Затраты на ремонт'] = self.data.loc[:].apply(
            lambda x: x['repairs_length'] * self.rep_price(x['D'])/1000, axis=1)
        self.data.loc[:, 'Затраты на ремонт (восстановленный)'] = self.data.loc[:].apply(
            lambda x: x['synthetic_length'] * self.rep_price(x['D'])/1000, axis=1)
        self.data.loc[:,['Ущерб предотвр. отказов, млн.руб.','Ущерб непредотвр. отказов, млн.руб.','Стоимость надежности, млн.руб.','Стоимость надежности']]=np.nan
        self.data.rename(columns={'length':"Длина сегмента, м","repairs_length":"Объемы Ремонта,м","synthetic_length":"Объемы Ремонта,м (восстановленные)",
                                  "n_predicted":"Число отказов","single_mask":"Первичный отказ","predicted_mask":"Предотвращенный отказ","lost_mask":"Пропущенный отказ"},inplace=True)
        self.data.loc[:,'Первый отказ на трубопроводе']=0
        i=self.data.loc[:,'Дата аварии'].argmin()
        self.data.at[self.data.index[i],'Первый отказ на трубопроводе']=1
        self.data.loc[:,'by']=np.nan
        for i in self.data.index:
            p = self.data.at[i, 'Число отказов']
            if not (p > 0):
                continue
            a = self.data.at[i, 'lbound']
            b = self.data.at[i, 'rbound']
            t = self.data.at[i, 'Дата аварии']
            mask = (self.data.loc[:, 'Адрес от начала участка'] >= a) & (self.data.loc[:, 'Адрес от начала участка'] <= b) & (
                        self.data.loc[:, 'Дата аварии'] > t)
            self.data.loc[mask,'by'] = i





        #'Адрес от начала участка' 'Наработка до отказа'
        #print(raw_length,synthetic_length)
        #self.data['repairs_length'].iloc[0]=raw_length
        #self.data['synthetic_length'].iloc[0]=synthetic_length

        #return self.data


