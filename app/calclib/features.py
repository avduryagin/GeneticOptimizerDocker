from app.calclib.engineering import features as Feat
from app.calclib.engineering import restrictions
from app.calclib import sets_methods as sm
from numpy.lib import recfunctions as rfn
import pandas as pd
from app.calclib.generator import ClRe
import os
import pickle
import numpy as np




class SVRfeatures(Feat):
    def __init__(self):
        super().__init__()
        self.reg_features = ['ads','tau']
        self.columns=['Наработка до отказа(new), лет', 'Адрес от начала участка (new)',
                            'Обводненность', 'getout','L,м', 'S','new_id','Дата аварии','index','to_out','D']
        self.cl_features = ['ads', 'ads05', 'ads1', 'ads2', 'ads3',
                            'ivl0', 'ivl1', 'ivl2', 'ivl3', 'ivl4', 'ivl5', 'nivl0', 'nivl1',
                            'nivl2', 'nivl3', 'nivl4', 'nivl5', 'wmean', 'amean', 'percent', 'tau',
                            'water', 'length']
        self.size_func = lambda x: 100 if x > 219 else 150

    def fit(self,xdata=pd.DataFrame([]),ident='new_id', expand=False, ints=np.array([100],dtype=np.int32),
            date=np.array([3],dtype=np.int32), steps=15, epsilon=1/12.,norm=True,mode='bw',restricts=True,drift=0.,clnorm=np.array([],dtype=np.int32),regnorm=np.array([],dtype=np.int32),**kwargs):

        def apply_norm(x=np.array([]), s=np.array([]), xindex=np.array([]), yindex=np.array([])):
            if yindex.shape[0] > 0:
                if xindex.shape[0] == 0:
                    x[:, yindex] = np.divide(x[:, yindex], s.reshape(-1, 1))

                else:
                    x[xindex, yindex] = np.divide(x[xindex, yindex], s.reshape(-1, 1))



        self.ident=ident
        self.expand=expand
        self.ints=ints
        self.date=date
        self.steps=steps
        self.epsilon=epsilon
        self.raw=xdata
        if len(self.reg_features)==0:
            self.reg_features = [str(x) for x in np.arange(self.steps)]
        #создание точек
        data=self.get_binary(self.raw,self.columns,date=self.date, ident=self.ident,expand=self.expand,ints=self.ints,steps=self.steps,epsilon=self.epsilon,mode=mode,restricts=restricts,drift=drift,size_func=self.size_func)

        if len(data)>0:
            self.data=np.vstack(data[0])
            self.cl=rfn.structured_to_unstructured(self.data[self.cl_features],dtype=np.float32).reshape(-1,len(self.cl_features))
            self.reg = rfn.structured_to_unstructured(self.data[self.reg_features], dtype=np.float32).reshape(-1,len(self.reg_features))
            #self.time_series = data[:, 1]
            self.time_series = np.array(data[1],dtype=object)
            self.s=self.data['s'].reshape(-1)
            self.top=self.data['top'].reshape(-1)
            self.shape = self.data['shape'].reshape(-1)
            self.horizon =self.data['horizon'].reshape(-1)
            self.features=self.data.dtype.names
            if norm:
                self.top=self.top/self.s
                self.horizon=self.horizon/self.s
            apply_norm(self.cl,yindex=clnorm,s=self.s)
            apply_norm(self.reg,yindex=regnorm,s=self.s)
                #self.reg = np.divide(self.reg,self.s.reshape(-1,1))

            self.ClRe = ClRe(c=self.cl, r=self.reg, s=self.s, t=self.time_series, shape=self.shape)

    def get_binary(self,xdata,columns,ident='ID простого участка',sortby='Наработка до отказа', expand=False, ints=np.array([100]), date=np.array([3]), steps=15, epsilon=1/12.,mode='bw',restricts=True,drift=0.,size_func=None):
        #mode - тип индексации
        xdata.sort_values(by=sortby, inplace=True)
        aggdata = xdata.groupby(ident)
        L = []
        for i, group in enumerate(aggdata):
            Length = group[1]['L'].iloc[0]
            D=group[1]['D'].iloc[0]
            if size_func is not None:
                size=size_func(D)
                ints=np.array([size])
            npints = np.array(ints) * 2
            data = group[1][['Адрес от начала участка','Наработка до отказа']].values
            mask = np.where(npints <= Length)[0]
            k = mask.shape[0]
            if k>0:
                for teta in ints:
                    if restricts:
                        restr=restrictions()
                        restr.fit(x=data,rep=group[1],enter=group[1]['Дата ввода'].min(),size=teta,length=Length,mode=mode)
                        val=restr.cover
                    else:
                        val=sm.cover(data,mode=mode,length=Length,size=teta,c0=0,c1=1)
                    index=group[1].index[val['i']]
                    subgroups=group[1].groupby('new_id')
                    uniques=np.unique(group[1].loc[index]['new_id'].values)

                    for ID in uniques:
                        subgroup=subgroups.groups[ID]
                        #length,x=group[1].iloc[i][['L,м','Адрес от начала участка']]
                        length=group[1].loc[subgroup[0],'L,м']
                        mask= np.where(npints <= length)[0]
                        n = mask.shape[0]
                        if n>0:
                            subindex=np.where(np.isin(subgroup,index))[0]
                            #sub - x,tau координаты ID
                            sub=group[1].loc[subgroup,columns].values
                            #максимальное значение возраста аварии
                            taumax=sub[:,0].max()
                            low=taumax*(1-drift)
                            drift_mask=sub[:,0]>=low
                            masked=sm.masked(sub[:,0],drift_mask,taumax)

                            for j in subindex:
                                s=subgroup[j]
                                v=index.get_loc(s)
                                x=group[1].loc[s,'a']
                                if teta * 2 <= length:
                                    a_=val['a'][v]-x
                                    b_=val['b'][v]-x
                                    #a_,b_=val[v][[1,2]]-x
                                    bound=sm.intersection((a_, b_), (0, length), shape=2).reshape(-1)
                                    if bound.shape[0]>0:
                                        if bound[1]-bound[0]>=teta:
                                            for d in date:
                                                tensor,ts = self.get_identity(sub, date=d, a=bound[0], b=bound[1], index=j,
                                                                            interval=teta,masked=masked)
                                                if tensor is not None:
                                                    L.append((tensor,ts))
                                                else:
                                                    print('empty id', group[0])

        transpose=[list(x) for x in zip(*L)]
        return transpose

    def get_identity(self,data, date=1, a=0, b=1, index=None, interval=100,masked=None):

        types = dict(
            names=['new_id', 'index', 'period', 'shape', 'Дата аварии', 'L,м', 'a', 'b', 'target', 'count', 'next',
                   'delta_next', 'delta',
                   'ads', 'ads05', 'ads1', 'ads2', 'ads3', 'ivl0', 'ivl1', 'ivl2', 'ivl3', 'ivl4', 'ivl5', 'nivl0',
                   'nivl1', 'nivl2', 'nivl3', 'nivl4', 'nivl5', 'wmean', 'amean', 'percent', 'tau', 'interval',
                   'water', 'x', 's','d', 'to_out', 'length', 'top', 'horizon'],
            formats=['U25', np.int32, np.int8, np.int32, 'datetime64[s]', np.float64, np.float64, np.float64, np.float64,
                     np.float64,np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64,
                     np.float64, np.float64, np.float64,np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64,
                     np.float64])

        def get_horizontal_counts(data=np.array([]), interval=100, L=100):
            mask = np.ones(data.shape[0], dtype=bool)
            intervals = []
            i = 0
            while mask.shape[0] > 0:
                y = data[-1]
                a = y - interval
                b = y + interval
                if a < 0:
                    a = 0
                if b > L:
                    b = L
                res = np.array([a, b])
                if i == 0:
                    intervals.append((0, 0, 0))
                    i = i + 1
                for ivl in intervals:
                    if res.shape[0] > 0:
                        res = sm.residual(res, ivl, shape=2).reshape(-1)
                if res.shape[0] > 0:
                    submask = (data >= res[0]) & (data <= res[1])
                    res = np.append(res, submask[submask == True].shape[0])
                    intervals.append(res)
                    data = data[~submask]
                    mask = mask[~submask]
                else:
                    mask[-1] = False
                    data = data[mask]
                    mask = mask[mask]
            return np.array(intervals[1:])

        if index is None:
            return None

        dtype = np.dtype(types)
        identity = np.empty(shape=(1), dtype=dtype)
        step = dict({'ads05': 0.5, 'ads1': 1., 'ads2': 2., 'ads3': 3.})
        tau = data[index, 0]
        x = data[index, 1]
        out = data[index, 3]
        length = data[index, 4]
        s = data[index, 5]
        id = data[index, 6]
        adate = data[index, 7]
        i = data[index, 8]
        to_out = data[index, 9]
        d=data[index, 10]
        identity['new_id'] = id
        identity['s'] = s
        identity['d'] = d
        identity['to_out'] = to_out
        identity['tau'] = tau
        identity['interval'] = interval
        identity['index'] = i
        identity['period'] = date
        identity['Дата аварии'] = adate
        identity['water'] = data[index, 2]
        identity['L,м'] = length
        identity['a'] = a
        identity['b'] = b
        identity['length'] = b - a
        identity['x'] = x
        identity['top'] = min(tau + date, tau + to_out)
        identity['horizon'] = tau + date

        if masked is not None:
            mtau = masked(index)
        else:
            mtau=tau
        mask = data[:, 0] <= tau
        hormask = mask
        if mtau > tau:
            hormask = data[:, 0] <= mtau

        identity['shape'] = hormask[hormask].shape[0]
        mask1 = (data[:, 1] >= a) & (data[:, 1] <= b)
        xmask = mask1 & mask
        ads = xmask[xmask == True].shape[0]
        dt = np.nan
        prev = 0
        if ads > 1:
            prev = data[xmask, 0][-2]

        dt = tau - prev
        identity['delta'] = dt
        identity['ads'] = ads

        # sparsed = sparse(data[:, 0][xmask], epsilon=epsilon)[-steps:]
        # for t in np.arange(1, steps + 1):
        # if -t >= -sparsed.shape[0]:
        # identity[columns[-t]] = sparsed[-t]
        # else:
        # identity[columns[-t]] = 0

        for k in step.keys():
            #dlt = tau - step[k]
            substep = data[:, 0] >= tau - step[k]
            smask = substep & xmask
            identity[k] = smask[smask == True].shape[0]
        ivls = get_horizontal_counts(data[:, 1][hormask], interval=interval, L=length)
        res = ivls[:, 1] - ivls[:, 0]
        identity['percent'] = res.sum() / length
        w_mean = data[:, 2][mask].mean()
        a_mean = data[:, 0][mask].mean()
        identity['wmean'] = w_mean
        identity['amean'] = a_mean
        ivl_counts = ivls[:, 2].astype(int)
        for ii in np.arange(6):
            if ii == 5:
                mask3 = ivl_counts >= ii + 1
                mask4 = ivl_counts >= 0
            else:
                mask3 = ivl_counts == ii + 1
                mask4 = ivl_counts <= ii + 1
            identity['ivl' + str(ii)] = mask3[mask3 == True].shape[0]
            identity['nivl' + str(ii)] = mask4[mask4 == True].shape[0]
        tmask = mask1 & (~mask)
        top = tau + date
        mask2 = (data[:, 0] <= min(top,out))
        ymask = tmask & mask2
        target = np.nan
        next = np.nan
        delta = np.nan

        identity['next'] = next
        identity['delta_next'] = delta
        # dic = {0: 8. / 12., 1: 7. / 12., 2: 5. / 12., 3: 4. / 12., 4: 3. / 12., 5: 3. / 12, 6: 2. / 12.,
        # 7: 2. / 12., }
        count = ymask[ymask].shape[0]
        if count > 0:
            inext = np.argmin(data[tmask, 0])
            # arange = np.arange(tmask.shape[0])
            # inext = arange[tmask][0]
            next = data[tmask, 0][inext]
            delta = next - tau
            identity['next'] = next
            identity['delta_next'] = delta

        if top <= out:
            if count > 0:
                target = 1
            else:
                target = 0
        else:
            if count > 0:
                target = 1
            else:
                target = np.nan
                count = np.nan

        identity['target'] = target
        identity['count'] = count
        ts = data[:, 0][xmask].astype(float)
        return identity, ts
        # return identity, sparsed

class SVRsingle(SVRfeatures):
    def __init__(self):
        super().__init__()

    def fit(self, xdata=pd.DataFrame([]), index=0,lbound=0,rbound=1,expand=False, ints=np.array(100, dtype=np.int32),
            date=np.array(3, dtype=np.int32),  norm=True, clnorm=np.array([], dtype=np.int32), regnorm=np.array([], dtype=np.int32), **kwargs):

        def apply_norm(x=np.array([]), s=np.array([]), xindex=np.array([]), yindex=np.array([])):
            if yindex.shape[0] > 0:
                if xindex.shape[0] == 0:
                    x[:, yindex] = np.divide(x[:, yindex], s.reshape(-1, 1))

                else:
                    x[xindex, yindex] = np.divide(x[xindex, yindex], s.reshape(-1, 1))


        self.expand = expand
        self.ints = ints
        self.date = date
        self.raw = xdata
        self.lbound=lbound
        self.rbound=rbound
        self.index=index
        if len(self.reg_features) == 0:
            self.reg_features = [str(x) for x in np.arange(self.steps)]
        # создание точек
        data = self.get_f(self.raw, self.index,self.lbound,self.rbound, self.columns, date=self.date, expand=self.expand,teta=self.ints)
        if data is None:
            print(self.index,self.lbound,self.rbound, self.columns, self.date, self.expand,self.ints)

        if len(data) > 0:
            self.data = np.vstack(data[0])
            self.cl = rfn.structured_to_unstructured(self.data[self.cl_features], dtype=np.float32).reshape(-1,
                                                                                                            len(self.cl_features))
            self.reg = rfn.structured_to_unstructured(self.data[self.reg_features], dtype=np.float32).reshape(-1,
                                                                                                              len(self.reg_features))
            # self.time_series = data[:, 1]
            self.time_series = np.array(data[1], dtype=object)
            self.s = self.data['s'].reshape(-1)
            self.top = self.data['top'].reshape(-1)
            self.shape = self.data['shape'].reshape(-1)
            self.horizon = self.data['horizon'].reshape(-1)
            self.features = self.data.dtype.names
            if norm:
                self.top = self.top / self.s
                self.horizon = self.horizon / self.s
            apply_norm(self.cl, yindex=clnorm, s=self.s)
            apply_norm(self.reg, yindex=regnorm, s=self.s)
            # self.reg = np.divide(self.reg,self.s.reshape(-1,1))

            self.ClRe = ClRe(c=self.cl, r=self.reg, s=self.s, t=self.time_series, shape=self.shape)

    def get_f(self,xdata,index,lbound,rbound,columns,sortby='Наработка до отказа', expand=False, teta=np.array(100,dtype=np.int32), date=np.array(3,dtype=np.int32)):
        #mode - тип индексации
        xdata.sort_values(by=sortby, inplace=True)
        npints = np.array(teta) * 2
        length = xdata.iloc[0]['L,м']
        assert lbound<rbound,'Заданы некорректные границы интервала a={0},b={1}'.format(lbound,rbound)
        bound = sm.intersection((lbound, rbound), (0, length), shape=2).reshape(-1)
        assert bound.shape[0]>0,"Заданы границы вне пределов простого участка"
        #li=bound[1]-bound[0]
        #if li<teta:
            #return None
        data=xdata[columns].values
        j=xdata.index.get_loc(index)
        L = []
        tensor, ts = self.get_identity(data, date=date, a=bound[0], b=bound[1], index=j,
                                       interval=teta)
        L.append((tensor, ts))
        transpose=[list(x) for x in zip(*L)]
        return transpose

#возвращает итератор индексов массива. Решение о исключении индекса принимается функцие принятия решений.
#входные данные: data=nd.array, shape=(n,7)
#data[:,0]-время >=0,data[:,1]-координата точки объемлющего отрезка >=0,data[:,2]- длина объемлющего отрезка
#data[:,3]-время в координатах вложенного отрезка >=0,data[:,4]-координата точки вложенного отрезка >=0,data[:,5]- длина вложенного отрезка
#data[:,6]-координата начала вложенного отрезка

class cover:
    def __init__(self,data,size=100):
        self.data=data
        self.size=size
        self.mask=np.ones(shape=data.shape[0],dtype=bool)
        self.marked = np.zeros(shape=data.shape[0], dtype=bool)
        self.predicted = np.zeros(shape=data.shape[0], dtype=bool)
        self.sa=np.arange(data.shape[0])
        self.count=data.shape[0]
        if data.shape[0]>1:
            self.sa=np.argsort(data[:,0])
            self.data=self.data[self.sa,:]
        self.index=0
        self.a=0
        self.b=0
        self.t=0
        self.values=[]


    def value(self):
        if (self.index<self.data.shape[0])&(self.count>0):
            x=self.data[self.index,1]
            self.t=self.data[self.index,0]
            start=self.data[self.index,6]
            l=self.data[self.index,5]
            end=start+l

            bound = sm.get_bounds(start=start,end=end,size=self.size,constrains=np.array(self.values))
            self.a,self.b=bound(x)
            return self.sa[self.index], self.a, self.b
        else: return None

    def get_index(self,state=True):
        imask=(self.data[:,1]<=self.b)&(self.data[:,1]>=self.a)
        dmask=(self.data[:,0]>=self.t)
        dmask[self.index]=False
        marked=imask&dmask
        self.marked[marked]=True
        if (state)&(self.count>0):
            mask=marked
            #mask=(self.data[:,1]<=self.b)&(self.data[:,1]>=self.a)&(self.data[:,0]>=self.t)
            self.mask[mask]=False
            self.predicted[mask]=True
            self.values.append([self.a, self.b])

        self.mask[self.index]=False
        indices = np.where(self.mask)[0]
        self.count = indices.shape[0]
        if self.count > 0:
            self.index = indices[0]

















