from __future__ import division
import warnings
import itertools
import string
from math import floor
from random import sample
from sympy.utilities.iterables import multiset_partitions as set_partitions


class _MeaningComponent():
    """
    This is a private base class 
    """
    def __init__(self, size):
        # check value
        self.size = size
        self._meanings = set([str(i) for i in list(xrange(size))]) # meanings are vectors of integers and graph nodes
        self._schemata = self._meanings | set('*') 

        ## THESE WEIGHTS ARE FOR THE SMITH-KIRBY WEIGHTS FOR PRODUCTION AND RECEPTION
        weights = list([1.0] * len(self._meanings)) + list([0.0])
        self._weights  = dict(zip((list(self._meanings)+list('*')),weights))        


    def meanings(self):
        return self._meanings

    def schemata(self):
        return self._schemata

    def weights(self):
        return self._weights

class OrderedMeaningComponent (_MeaningComponent):
    """
    These meaning components implement lattice-like meaning structures
    that represent naturally ordered meanings such as quantity,
    magnitude and relative degree. These were introduced by the
    original Smith-Brighton-Kirby ILM models of early 2000s.

    In ILMpy, generalization in ordered components occurs along
    lattice dimensions across the component, as in the original ILM
    models. This generalization operator is denoted with the
    asterisk(*) wildcard character in Smith 2003a technical report,
    Brighton et al. (2005) and so on.
    
    >>> omc = OrderedMeaningComponent(5)
    >>> omc.generalize(4)
    ['*']
    >>> omc.meanings()
    set(['1', '0', '3', '2', '4'])
    >>> omc.schemata()
    set(['1', '0', '3', '2', '4', '*'])
    >>> omc.weights()
    {'*': 0.0, '1': 1.0, '0': 1.0, '3': 1.0, '2': 1.0, '4': 1.0}
    """    
    def __init__(self, size):
        _MeaningComponent.__init__(self,size)

    def generalize(self, meaning):
        return  ['*']
        
class UnorderedMeaningComponent (_MeaningComponent):    
    """
    These meaning components represent set-like meaning structures
    representing a collection of meanings so distinct, they cannot be
    generalized. These are introduced with ILMpy.
    
    >>> umc = UnorderedMeaningComponent(5)
    >>> umc.generalize(4)
    [4]
    >>> umc.meanings()
    set(['1', '0', '3', '2', '4'])
    >>> umc.schemata()
    set(['1', '0', '3', '2', '4', '*'])
    >>> umc.weights()
    {'*': 0.0, '1': 1.0, '0': 1.0, '3': 1.0, '2': 1.0, '4': 1.0}
    """
    def __init__(self, size):
        _MeaningComponent.__init__(self,size)
    
    def generalize(self, meaning):
        return [meaning]; # the generalization identity 

class _MeaningSpace():
    """
    This is a private base class 
    """
    def __init__(self):
        self._meanings = None
        self._schemata = None
        self._weights  = None

class CombinatorialMeaningSpace (_MeaningSpace):
    """
    >>> meaning_space = CombinatorialMeaningSpace()
    >>> meanings1    = OrderedMeaningComponent(3)
    >>> meanings2    = UnorderedMeaningComponent(2)
    >>> meanings3    = OrderedMeaningComponent(2)
    
    >>> meaning_space.add_component(meanings1)
    >>> meaning_space.add_component(meanings2)
    >>> meaning_space.add_component(meanings3)

    >>> set(meaning_space.generalize('111'))
    set(['*1*', '11*', '111', '*11'])

    >>> list(meaning_space.analyze('111',2))
    [['*11', '11*'], ['*1*', '111'], ['*11', '11*']]

    >>> list(meaning_space.analyze('111',3))
    [['*11', '111', '11*']]

    >>> meaning_space.meanings()
    ['111', '110', '101', '100', '011', '010', '001', '000', '211', '210', '201', '200']

    >>> meaning_space.schemata()
    ['111', '110', '11*', '101', '100', '10*', '1*1', '1*0', '1**', '011', '010', '01*', '001', '000', '00*', '0*1', '0*0', '0**', '211', '210', '21*', '201', '200', '20*', '2*1', '2*0', '2**', '*11', '*10', '*1*', '*01', '*00', '*0*', '**1', '**0', '***']

    >>> meaning_space.sample(10)

    
    """
    def __init__(self):
        _MeaningSpace.__init__(self)
        self.components = []
        self._weights = {}
        self.length = 0

    def add_component(self,component):
        self.components.append(component)
        self.length += 1
        meanings = []
        schemata = []
        keys     = []
        weights  = []
        for component in self.components:
            meanings.append(component.meanings())
            schemata.append(component.schemata())
            keys.append(component.weights().keys())
            weights.append(component.weights().values())
            
        self._meanings = [''.join(s) for s in itertools.product(*meanings) ]
        self._schemata = [''.join(s) for s in itertools.product(*schemata) ]
        self._weights  = dict(zip(map(''.join,itertools.product(*keys)),map(sum,itertools.product(*weights))))

    def meanings(self):
        return self._meanings

    def schemata(self):
        return self._schemata

    def weights(self,schema):
        if (schema in self._weights):
            return (self._weights[schema] / self.length)
        else:
            None

    def analyze(self,meaning, length):
        mlist = list(meaning)
        partitions = set_partitions(range(len(meaning)),length)
        for partition in partitions:
            analysis = []
            for iset in partition:
                rlist = mlist[:]
                for i in iset:
                    rlist[i] = self.components[i].generalize(rlist[i])[0]
                analysis.append(''.join(rlist))    
            yield analysis

    def generalize(self,meaning):
        for i in range(len(meaning)):
            for locs in itertools.combinations(range(len(meaning)), i):
                meanings = [[component] for component in meaning]
                for loc in locs:
                    original_meaning = meaning[loc]
                    meanings[loc] = self.components[loc].generalize(original_meaning)
                for chars in itertools.product(*meanings):
                    schema = ''.join(chars)
                    yield schema 
    
    def sample(self,number):
        if (number < 0 or (number != floor(number))):
            raise ValueError("Parameter number must be an integer >= 0. You passed %f" % (number))
        return sample(self._meanings,number)
        

if __name__ == "__main__":
    import doctest
    doctest.testmod()
   
