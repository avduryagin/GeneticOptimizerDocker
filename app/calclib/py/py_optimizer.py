import numpy as np
from numpy import random as rd

class Individual:
    def __init__(self,size=1,bound=0.):
        assert size>0,"size must be a positive number"
        #assert bound > 0, "bound must be a positive float"
        self.val=0.
        self.rest=0.
        self.bound=bound
        self.size=size
        self.code=np.zeros(size,dtype=bool)

    def fit(self,x=np.array([]),y=np.array([])):
        assert (x.shape[0]==self.size)&(y.shape[0]==self.size), "x, y must be an equal shape arrays with shape= self.size"
        index=np.arange(self.size)
        shuffled=rd.permutation(index)
        i=0
        s=0.
        while i<self.size:
            j=shuffled[i]
            s+=abs(x[j])
            if s>self.bound:
                break
            self.code[j]=True
            i+=1
        return

    def value(self,x=np.array([]),y=np.array([])):
        i=0
        self.val=0.
        self.rest=0.
        v=0.
        while i<self.size:
            self.rest+=self.code[i]*x[i]
            v+=(~self.code[i])*y[i]
            i+=1
        self.val=v+self.rest
        return self.val

    def mutate(self,ncell=1):
        i=0
        while i<ncell:
            j=rd.randint(self.size)
            self.code[j]=~self.code[j]
            i+=1


class Population:

    def __init__(self,size=10,bound=0.,x=np.array([]),y=np.array([]),threshold=0.7,epsilon=1e-3,mutate_cell=1,tolerance=0.7,cast_number=3):
        assert size > 0, "size must be a positive number"
        self.size = size
        #assert bound > 0, "bound must be a positive float"
        self.bound=bound
        assert (y.shape[0] == x.shape[0]), "x, y must be an equal shape"
        self.length=x.shape[0]
        self.threshold=threshold
        self.tolerance=tolerance
        self.epsilon=epsilon
        self.mutate_cell=mutate_cell
        self.cast_number=cast_number
        self.population=dict()
        self.x=x
        self.y=y
        self.minval=np.inf
        self.maxval=np.NINF
        self.optimal_individ=None
        self.maxiter=1000
        self.epoch=1000
        self.niter=0
        self.log=[]
        self.solution=None
        self.cross_method=self.cast

        i=0
        while i<size:
            individ = Individual(size=x.shape[0], bound=self.bound)
            individ.fit(x,y)
            val=individ.value(x,y)
            if val<self.minval:
                self.minval=val
                self.optimal_individ=i
            if val>self.maxval:
                self.maxval=val
            self.population[i]=individ
            i+=1
        self.log.append(self.minval)
        self.solution=self.population[self.optimal_individ]

    def __call__(self,index=0):
        try:
            individ=self.population[index]
            return individ
        except KeyError:
            return None
    def cross(self,iparent0,iparent1):
        try:
            parent0 = self.population[iparent0]
            parent1 = self.population[iparent1]
            index=rd.randint(self.length)
            child=Individual(size=self.length,bound=self.bound)
            child.code[:index]=parent0.code[:index]
            child.code[index:] = parent1.code[index:]
            child.mutate(self.mutate_cell)
            val=child.value(self.x,self.y)
            del val
            if abs(child.rest)>self.bound:
                return None
            return child
        except KeyError:
            return None
    def entanglement(self,iparent0,iparent1):
        def cross(a=np.array([],dtype=bool),b=np.array([],dtype=bool)):
            c=b.copy()
            i=0
            while(i<c.shape[0]):
                c[i]=a[i]
                i+=2
            return c
        try:
            parent0 = self.population[iparent0]
            parent1 = self.population[iparent1]

            #index=rd.randint(self.length)
            child=Individual(size=self.length,bound=self.bound)
            a=parent0.code
            b=parent1.code
            alpha=rd.random()
            if alpha<0.5:
                child.code=cross(a,b)
            else:
                child.code = cross(b, a)

            child.mutate(self.mutate_cell)
            val=child.value(self.x,self.y)
            del val
            if abs(child.rest)>self.bound:
                return None
            return child
        except KeyError:
            return None
    def select_population(self):
        individ_list=[]
        delta=(self.maxval-self.minval)*self.threshold
        for k in self.population.keys():
            if not (self.population[k].val>self.minval+delta):
                individ_list.append(k)
        return np.array(individ_list,dtype=np.int32)

    def cast(self,indices=np.array([],dtype=np.int32)):
        try:
            index=rd.randint(indices.shape[0],size=self.cast_number)
            minval=np.inf
            optimal_index=None
            for i in index:
                j=indices[i]
                v=self.population[j].val
                if v<minval:
                    minval=v
                    optimal_index=j
            return optimal_index
        except KeyError:
            return None

    def get_new_population(self):
        population=dict()
        indices=self.select_population()
        maxval=np.NINF
        minval=np.inf
        optimal_index=None
        k=0
        i=0
        while (k<self.size)&(i<self.maxiter):
            parent0=self.cast(indices)
            parent1=parent0
            if indices.shape[0]>1:
                while (parent0==parent1):
                    parent1 = self.cast(indices)

            i+=1
            if rd.random()<self.tolerance:
                individ=self.entanglement(parent0,parent1)
                if individ is not None:
                    population[k]=individ
                    if individ.val<minval:
                        minval=individ.val
                        optimal_index=k
                    if individ.val>maxval:
                        maxval=individ.val
                    k+=1
        return population,optimal_index,minval, maxval

    def optimize(self,allow_count=3):
        self.niter=0
        main_val = self.minval
        #count=0
        while self.niter<self.epoch:
            count=0
            local_val=np.inf
            local_individ=None
            while count<allow_count:
                population, optimal_index, minval, maxval = self.get_new_population()
                self.population = population
                self.minval = minval
                self.maxval = maxval
                self.optimal_individ = optimal_index
                self.log.append(self.minval)
                if (minval<local_val)&(abs(minval-local_val)>self.epsilon):
                    local_val=minval
                    local_individ=self.population[optimal_index]
                count+=1
                self.niter+=1
                #print("niter {0} local_val {1:.2f} sol_val {2:.2f} main_val {3:.2f}".format(self.niter,local_val,self.solution.val,main_val))
            if (local_val<main_val)&(abs(local_val-main_val)>self.epsilon):
                main_val=local_val
                self.solution=local_individ
            else:
                break