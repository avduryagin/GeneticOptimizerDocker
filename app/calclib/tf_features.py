import numpy as np
import app.calclib.sets_methods as sm
from app.calclib.features import SVRfeatures


class TFFeatures(SVRfeatures):
    def __init__(self):
        super().__init__()
        self.types=dict(
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
        self.dtypes=np.dtype(self.types)
        cl_types = dict({'names': self.cl_features})
        cl_types['formats'] = [self.dtypes[k] for k in cl_types['names']]
        self.cl_dtypes=np.dtype(cl_types)
        self.top=np.array([],dtype=np.float32)


    def empty_vector(self,dtype):
        empty=np.empty(shape=(1), dtype=dtype)
        return empty

    def fit(self,*args,**kwargs):
        super().fit(*args,**kwargs)
        self.groups = self.raw.groupby('new_id').groups

    def get_horizontal_counts(self,data=np.array([]), interval=100, L=100):
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

    def get_identity(self,data, date=1, a=0, b=1, index=None, interval=100,masked=None):
        if index is None:
            return None

        identity = self.empty_vector(self.dtypes)
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
        ads = xmask[xmask].shape[0]
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
            identity[k] = smask[smask].shape[0]
        ivls = self.get_horizontal_counts(data[:, 1][hormask], interval=interval, L=length)
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
            identity['ivl' + str(ii)] = mask3[mask3].shape[0]
            identity['nivl' + str(ii)] = mask4[mask4].shape[0]
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
        ts=np.array([tau],dtype=np.float32)
        #ts = data[:, 0][xmask].astype(float)
        return identity, ts

    def get_cl(self, data, a=0., b=1., index=None,interval=100.,length=100.):
        if index is None:
            return None

        identity = self.empty_vector(self.cl_dtypes)
        step = dict({'ads05': 0.5, 'ads1': 1., 'ads2': 2., 'ads3': 3.})
        tau = data[index, 0]
        x = data[index, 1]
       # length = data[index, 4]

        identity['tau'] = tau
        identity['water'] = data[index, 2]
        identity['length'] = b - a
        #identity['x'] = x
        mask = data[:, 0] <= tau
        hormask = mask
        #identity['shape'] = hormask[hormask].shape[0]
        mask1 = (data[:, 1] >= a) & (data[:, 1] <= b)
        xmask = mask1 & mask
        ads = xmask[xmask].shape[0]
        identity['ads'] = ads

        for k in step.keys():
            # dlt = tau - step[k]
            substep = data[:, 0] >= tau - step[k]
            smask = substep & xmask
            identity[k] = smask[smask].shape[0]
        ivls = self.get_horizontal_counts(data[:, 1][hormask], interval=interval, L=length)
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
            identity['ivl' + str(ii)] = mask3[mask3].shape[0]
            identity['nivl' + str(ii)] = mask4[mask4].shape[0]
        return identity
    def getnewcl(self,index=None,generated=np.array([]),a=0.,b=1.,interval=100.):
        def approach_wp(data,tau):
            k=0
            i=0
            mindelta=np.inf
            while i<data.shape[0]:
                tau_=data[i,0]
                if tau_>tau:
                    i+=1
                    continue
                delta=tau-tau_
                if delta<mindelta:
                    mindelta=delta
                    k=i
                i+=1
            return data[k,2]

        if index is None or generated.shape[0]==0:
            return None
        id_=self.raw.at[index,'new_id']
        length=self.raw.at[index,'L,м']
        indices=self.groups[id_]
        tau=generated[0,0]
        columns=self.columns[:3]
        raw=self.raw.loc[indices,columns].values
        wp=approach_wp(raw,tau)
        generated[0,2]=wp
        data=np.vstack([generated,raw])
        vector=self.get_cl(data,a=a,b=b,length=length,interval=interval,index=0)
        return vector




