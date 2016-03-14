from __future__ import division
import warnings
import itertools
import string
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

    def meanings(self):
        return self._meanings

    def schemata(self):
        return self._schemata

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
        self._all_general = None

class CombinatorialMeaningSpace (_MeaningSpace):
    """
    """
    def __init__(self):
        _MeaningSpace.__init__(self)
        self.components = []
        self.component_added = False
        self._weights = {}

    def weights(self,schema):
        if (schema in self._weights):
            return self._weights[schema]
        else:
            self.generalize(schema)
            return self._weights[schema]

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

    def add_component(self,component):
        self.components.append(component)
        self.component_added = True

    def generalize(self,meaning):
        for i in range(len(meaning)):
            for locs in itertools.combinations(range(len(meaning)), i):
                meanings = [[component] for component in meaning]
                for loc in locs:
                    original_meaning = meaning[loc]
                    meanings[loc] = self.components[loc].generalize(original_meaning)
                for chars in itertools.product(*meanings):
                    schema = ''.join(chars)
                    self._weights[schema] = (1.0 - (len(locs)/len(meaning)))
                    yield schema 

    def schemata(self):
        if self._schemata is None or self.component_added:
            all_schemata = []
            for component in self.components:
                all_schemata.append(component.schemata())
            self.component_added = False
            self._schemata = [''.join(m) for m in list(itertools.product(*all_schemata))]# if not m == self.all_general()]
        return self._schemata

    def meanings(self):
        if self._meanings is None or self.component_added:
            all_meanings = []
            for component in self.components:
                all_meanings.append(component.meanings())
            self.component_added = False
            self._meanings = [''.join(m) for m in list(itertools.product(*all_meanings))]# if not m == self.all_general()]
        return self._meanings


    ## def all_general(self):
    ##     if self._all_general is None or self.component_added:
    ##         all_general = []
    ##         for component in self.components:
    ##             all_general.append(component.generalize())
    ##         self.component_added = False
    ##         self._all_general = [''.join(m) for m in list(itertools.product(*all_general))][0]
    ##     return self._all_general 
    
    def sample(self,number):
        if (number < 0 or (number != floor(number))):
            raise ValueError("Parameter number must be an integer >= 0. You passed %f" % (number))
        return [choice(self.mea) for _ in xrange(4)]
        

if __name__ == "__main__":
    import doctest
    doctest.testmod()
   
