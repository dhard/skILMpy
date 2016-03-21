from __future__ import division
import warnings
import itertools
import string
import random
import copy
from distance import hamming
from itertools import chain, combinations
from collections import defaultdict
from sympy.utilities.iterables import multiset_partitions as set_partitions

class _SignalComponent():
    """
    This is a private base class 
    """
    def __init__(self, noiserate = 0):
        self._noiserate = noiserate
        self.noisy = False
        if (noiserate > 0):
            self.noisy = True
    
    def sounds(self):
        return self._sounds

    def schemata(self):
        return self._schemata

    def weights(self):
        return self._weights

    def get_noiserate(self):
        return self._noiserate

    ## This is the only mutable attribute
    def set_noiserate(self, noiserate):
        self._noiserate = noiserate



class SignalComponent (_SignalComponent):
    """
    >>> space      = SignalComponent(set('aeiou'))
    >>> space.sounds()
    set(['a', 'i', 'e', 'u', 'o'])
    >>> space.schemata()
    set(['a', 'e', 'i', 'u', '*', 'o'])
    >>> space.weights()
    {'a': 1.0, 'e': 1.0, 'i': 1.0, '*': 0.0, 'o': 1.0, 'u': 1.0}
    >>> space.distort('a')
    ['i', 'e', 'u', 'o']
    >>> space.distort('u')
    ['a', 'i', 'e', 'o']
    """    
    def __init__(self, sounds, noiserate = 0):
        _SignalComponent.__init__(self, noiserate)
        self._sounds = sounds
        self._schemata = self.sounds() | set('*')

        ## THESE WEIGHTS ARE FOR THE SMITH-KIRBY WEIGHTS FOR PRODUCTION AND RECEPTION
        weights = list([1.0] * len(self._sounds)) + list([0.0])
        self._weights  = dict(zip((list(sounds)+list('*')),weights))

    def generalize(self, sound):
        return ['*']

    def distort(self, sound):
        distortions = self._sounds.copy()
        distortions.remove(sound) 
        return list(distortions)

class TransformSignalComponent (_SignalComponent):    
    """
    >>> transform      = TransformSignalComponent('ae','AE')
    >>> transform.shortsounds
    'ae'
    >>> transform.longsounds
    'AE'
    >>> transform.sounds()
    set(['a', 'A', 'e', 'E'])
    >>> transform.schemata()
    set(['a', 'A', '#', 'e', '@', 'E'])
    >>> transform.weights()
    {'a': 1.0, 'A': 1.0, '#': 0.0, 'e': 1.0, '@': 0.0, 'E': 1.0}
    """
    def __init__(self, shortsounds, longsounds, noiserate = 0):
        _SignalComponent.__init__(self, noiserate)
        if (len(shortsounds) != len(longsounds)):
            raise ValueError("Arguments to initialize TransformSignalComponent must be of equal length. You passed %s and %s" % (shortsounds,longsounds))
        if (len(shortsounds) > 12):
            raise ValueError("Only up to 12 transformable sound-pairs are supported. You passed %u" % (len(shortsounds)))
        self.shortsounds = shortsounds
        self.longsounds  = longsounds
        shortlong = shortsounds + longsounds
        longshort = longsounds + shortsounds
        self.translation_table = string.maketrans(shortlong,longshort)

        transform_wildcards = list("@#!+?$&%=<>.")[:len(shortsounds)]
        
        self._generalizations = dict(zip(list(shortlong),(transform_wildcards * 2))) ## limited to 12

        self._sounds   = set(shortsounds) | set (longsounds)
        self._schemata = self._sounds | set(transform_wildcards)
        
        ## THESE WEIGHTS ARE FOR THE SMITH-KIRBY WEIGHTS FOR PRODUCTION AND RECEPTION
        weights = list([1.0] * len(self._sounds)) + list([0.0] * len(transform_wildcards))
        self._weights  = dict(zip((list(shortlong)+transform_wildcards),weights))

    def generalize(self, sound):
        return [self._generalizations[sound]]

    def distort(self, sound):
        return list(string.translate(sound,self.translation_table)) 

class _SignalSpace():
    """
    This is a private base class 
    """
    def __init__(self):
        pass

class WordSignalSpace (_SignalSpace):
    """
    WordSignalSpace models natural utterances with a finite number of discrete sounds,
    a finite length, generalizable transformations on sounds, and anisotropic noise.

    For word models, nu defines the base noise rate and may be any number greater or equal to 0.
    The base noise rate is multiplied by dimension-specific noise rates given in the input argument
    This defines the per-symbol noise rate per transaction. 
    The probability of no change of a symbol is defined as (1 - nu).

    >>> signal_space = WordSignalSpace()
    >>> sounds1      = SignalComponent(set('bp'))
    >>> sounds2      = SignalComponent(set('aeiou'))
    >>> sounds3      = SignalComponent(set('dt'))
    
    >>> signal_space.add_component(sounds1)
    >>> signal_space.add_component(sounds2)
    >>> signal_space.add_component(sounds3)

    >>> set(signal_space.generalize('bad'))
    set(['b*d', 'b**', 'bad', '*a*', '*ad', '**d', 'ba*'])

    >>> list(signal_space.analyze('bad',2))
    [['**d', 'ba*'], ['*a*', 'b*d'], ['*ad', 'b**']]

    >>> list(signal_space.analyze('bad',3))
    [['*ad', 'b*d', 'ba*']]

    >>> [[k,v] for k,v in signal_space.distort('bad')]
    [['bad', 1.0]]

    >>> sounds4      = TransformSignalComponent('ae','AE')
    >>> signal_space.add_component(sounds4)

    >>> set(signal_space.generalize('bada'))
    set(['*a*a', '*a*@', 'b*d@', 'b*da', '***a', '**d@', '**da', '*ada', '*ad@', 'b**@', 'bada', 'bad@', 'ba*a', 'ba*@', 'b**a'])

    >>> set(signal_space.generalize('badA'))
    set(['*a*A', '*a*@', 'b*d@', 'b*dA', '***A', '**d@', '**dA', '*adA', '*ad@', 'b**@', 'badA', 'bad@', 'ba*A', 'ba*@', 'b**A'])

    >>> signal_space.signals()
    ['pada', 'padA', 'pade', 'padE', 'pata', 'patA', 'pate', 'patE', 'pida', 'pidA', 'pide', 'pidE', 'pita', 'pitA', 'pite', 'pitE', 'peda', 'pedA', 'pede', 'pedE', 'peta', 'petA', 'pete', 'petE', 'puda', 'pudA', 'pude', 'pudE', 'puta', 'putA', 'pute', 'putE', 'poda', 'podA', 'pode', 'podE', 'pota', 'potA', 'pote', 'potE', 'bada', 'badA', 'bade', 'badE', 'bata', 'batA', 'bate', 'batE', 'bida', 'bidA', 'bide', 'bidE', 'bita', 'bitA', 'bite', 'bitE', 'beda', 'bedA', 'bede', 'bedE', 'beta', 'betA', 'bete', 'betE', 'buda', 'budA', 'bude', 'budE', 'buta', 'butA', 'bute', 'butE', 'boda', 'bodA', 'bode', 'bodE', 'bota', 'botA', 'bote', 'botE']

    >>> signal_space.schemata()
    ['pa*a', 'pa*A', 'pa*#', 'pa*e', 'pa*@', 'pa*E', 'pada', 'padA', 'pad#', 'pade', 'pad@', 'padE', 'pata', 'patA', 'pat#', 'pate', 'pat@', 'patE', 'pe*a', 'pe*A', 'pe*#', 'pe*e', 'pe*@', 'pe*E', 'peda', 'pedA', 'ped#', 'pede', 'ped@', 'pedE', 'peta', 'petA', 'pet#', 'pete', 'pet@', 'petE', 'pi*a', 'pi*A', 'pi*#', 'pi*e', 'pi*@', 'pi*E', 'pida', 'pidA', 'pid#', 'pide', 'pid@', 'pidE', 'pita', 'pitA', 'pit#', 'pite', 'pit@', 'pitE', 'pu*a', 'pu*A', 'pu*#', 'pu*e', 'pu*@', 'pu*E', 'puda', 'pudA', 'pud#', 'pude', 'pud@', 'pudE', 'puta', 'putA', 'put#', 'pute', 'put@', 'putE', 'p**a', 'p**A', 'p**#', 'p**e', 'p**@', 'p**E', 'p*da', 'p*dA', 'p*d#', 'p*de', 'p*d@', 'p*dE', 'p*ta', 'p*tA', 'p*t#', 'p*te', 'p*t@', 'p*tE', 'po*a', 'po*A', 'po*#', 'po*e', 'po*@', 'po*E', 'poda', 'podA', 'pod#', 'pode', 'pod@', 'podE', 'pota', 'potA', 'pot#', 'pote', 'pot@', 'potE', 'ba*a', 'ba*A', 'ba*#', 'ba*e', 'ba*@', 'ba*E', 'bada', 'badA', 'bad#', 'bade', 'bad@', 'badE', 'bata', 'batA', 'bat#', 'bate', 'bat@', 'batE', 'be*a', 'be*A', 'be*#', 'be*e', 'be*@', 'be*E', 'beda', 'bedA', 'bed#', 'bede', 'bed@', 'bedE', 'beta', 'betA', 'bet#', 'bete', 'bet@', 'betE', 'bi*a', 'bi*A', 'bi*#', 'bi*e', 'bi*@', 'bi*E', 'bida', 'bidA', 'bid#', 'bide', 'bid@', 'bidE', 'bita', 'bitA', 'bit#', 'bite', 'bit@', 'bitE', 'bu*a', 'bu*A', 'bu*#', 'bu*e', 'bu*@', 'bu*E', 'buda', 'budA', 'bud#', 'bude', 'bud@', 'budE', 'buta', 'butA', 'but#', 'bute', 'but@', 'butE', 'b**a', 'b**A', 'b**#', 'b**e', 'b**@', 'b**E', 'b*da', 'b*dA', 'b*d#', 'b*de', 'b*d@', 'b*dE', 'b*ta', 'b*tA', 'b*t#', 'b*te', 'b*t@', 'b*tE', 'bo*a', 'bo*A', 'bo*#', 'bo*e', 'bo*@', 'bo*E', 'boda', 'bodA', 'bod#', 'bode', 'bod@', 'bodE', 'bota', 'botA', 'bot#', 'bote', 'bot@', 'botE', '*a*a', '*a*A', '*a*#', '*a*e', '*a*@', '*a*E', '*ada', '*adA', '*ad#', '*ade', '*ad@', '*adE', '*ata', '*atA', '*at#', '*ate', '*at@', '*atE', '*e*a', '*e*A', '*e*#', '*e*e', '*e*@', '*e*E', '*eda', '*edA', '*ed#', '*ede', '*ed@', '*edE', '*eta', '*etA', '*et#', '*ete', '*et@', '*etE', '*i*a', '*i*A', '*i*#', '*i*e', '*i*@', '*i*E', '*ida', '*idA', '*id#', '*ide', '*id@', '*idE', '*ita', '*itA', '*it#', '*ite', '*it@', '*itE', '*u*a', '*u*A', '*u*#', '*u*e', '*u*@', '*u*E', '*uda', '*udA', '*ud#', '*ude', '*ud@', '*udE', '*uta', '*utA', '*ut#', '*ute', '*ut@', '*utE', '***a', '***A', '***#', '***e', '***@', '***E', '**da', '**dA', '**d#', '**de', '**d@', '**dE', '**ta', '**tA', '**t#', '**te', '**t@', '**tE', '*o*a', '*o*A', '*o*#', '*o*e', '*o*@', '*o*E', '*oda', '*odA', '*od#', '*ode', '*od@', '*odE', '*ota', '*otA', '*ot#', '*ote', '*ot@', '*otE']

    >>> signal_space.weights('padE')
    1.0
    >>> signal_space.weights('*ad@')
    0.5
    >>> signal_space.weights('***A')
    0.25

    >>> signal_space2 = WordSignalSpace()
    >>> sounds1       = SignalComponent(set('bpdr'),noiserate=0.1)
    >>> sounds1.distort('b')
    ['p', 'r', 'd']
    >>> sounds2       = TransformSignalComponent('aeiou','AEIOU')
    >>> signal_space2.add_component(sounds1)
    >>> signal_space2.add_component(sounds2)
    >>> [[k,v] for k,v in signal_space2.distort('ba')]
    [['ba', 0.9], ['pa', 0.03333333333333333], ['ra', 0.03333333333333333], ['da', 0.03333333333333333]]
    
    >>> sounds3       = SignalComponent(set('dt'))
    >>> signal_space2.add_component(sounds3)
    >>> [[k,v] for k,v in signal_space2.distort('bad')]
    [['bad', 0.9], ['pad', 0.03333333333333333], ['rad', 0.03333333333333333], ['dad', 0.03333333333333333]]

    >>> sounds4      = TransformSignalComponent('ae','AE', noiserate=0.2)
    >>> signal_space2.add_component(sounds4)
    >>> [[k,v] for k,v in signal_space2.distort('bada')]    
    [['bada', 0.7200000000000001], ['badA', 0.18000000000000002], ['pada', 0.02666666666666667], ['padA', 0.006666666666666667], ['rada', 0.02666666666666667], ['radA', 0.006666666666666667], ['dada', 0.02666666666666667], ['dadA', 0.006666666666666667]]
    """
    def __init__(self):
        _SignalSpace.__init__(self)
        self.length = 0
        self._components = []
        self._sounds = []
        self._signals = []
        self._schemata = []
        self._weightkeys = []
        self._weightvalues = []
        self._weights = {}
        self._noiserates = []
        self._hamming = defaultdict(dict)
        self.noisy = False
        
    def add_component(self,component):
        if (self.length == 0):
            self._signals = [''.join(s) for s in itertools.product(component.sounds()) ]
            self._schemata = [''.join(s) for s in itertools.product(component.schemata()) ]
            self._weightkeys  = [''.join(s) for s in itertools.product(component.weights().keys()) ]
            self._weightvalues  = [sum(s) for s in itertools.product(component.weights().values()) ]
            self._weights  = dict(zip(self._weightkeys,self._weightvalues))
        else:
            self._signals = [''.join(s) for s in itertools.product(self._signals,component.sounds()) ]
            self._schemata = [''.join(s) for s in itertools.product(self._schemata,component.schemata()) ]
            self._weightkeys  = [''.join(s) for s in itertools.product(self._weightkeys,component.weights().keys()) ]
            self._weightvalues  = [sum(s) for s in itertools.product(self._weightvalues,component.weights().values()) ]
            self._weights  = dict(zip(self._weightkeys,self._weightvalues))

        if (component.noisy):
            self.noisy = True 
        self.length += 1
        self._components.append(component)
        self._noiserates.append(component.get_noiserate())


    def components(self,i):
         return self._components[i]
            
    def signals(self):
        return self._signals

    def schemata(self):
        return self._schemata

    def weights(self,schema):
        if (schema in self._weights):
            return (self._weights[schema] / self.length)
        else:
            None

    def noiserates(self):
        return self._noiserates

    def hamming(self,sig1,sig2):
        assert len(sig1) == len(sig2)
        if (sig1 == sig2):
            return 0
        elif sig1 in self._hamming and sig2 in self._hamming[sig1]:
            return self._hamming[sig1][sig2]
        else:
            self._hamming[sig1][sig2] = self._hamming[sig2][sig1] = (hamming(sig1,sig2)/self.length)
            return self._hamming[sig1][sig2]

    def analyze(self, signal, length):
        slist = list(signal)
        partitions = set_partitions(range(len(signal)),length)
        for partition in partitions:
            analysis = []
            for iset in partition:
                rlist = slist[:]
                for i in iset:
                    rlist[i] = self.components(i).generalize(rlist[i])[0]
                analysis.append(''.join(rlist))    
            yield analysis

    def generalize(self,signal):
        for i in range(len(signal)):
            for locs in itertools.combinations(range(len(signal)), i):
                sounds = [[char] for char in signal]
                for loc in locs:
                    original_sound = signal[loc]
                    sounds[loc] = self.components(loc).generalize(original_sound)
                for chars in itertools.product(*sounds):
                    schema = ''.join(chars)
                    yield schema 

    def distort (self,signal):
        slist = list(signal)
        if self.noisy:
            rates = self.noiserates()
            noisyindices = [ i for i in xrange(len(signal)) if rates[i] > 0 ]
            dlist = [ self.components(i).distort(signal[i]) if i in noisyindices else [] for i in xrange(len(signal)) ]
            sfreq = [ (1 - rates[i]) if i in noisyindices else 1 for i in xrange(len(signal))]
            dfreq = [ (rates[i] / len(dlist[i])) if i in noisyindices else 1 for i in xrange(len(signal)) ]
            clist = [ [s] for s in signal ]
            for i in noisyindices:
                clist[i].extend(dlist[i])

            for chars in itertools.product(*clist):
                utterance = ''.join(chars)
                frequency = 1.0
                for i in noisyindices:
                    if (utterance[i] == slist[i]):
                        frequency *= sfreq[i]
                    else:
                        frequency *= dfreq[i]
                yield utterance, frequency

        else:
            yield signal, 1.0


    def compute_neighbors (self, signal, position):
        clist = [ [s] for s in signal ]
        clist[position] = self.components(position).distort(signal[position]) 
        for chars in itertools.product(*clist):
            utterance = ''.join(chars)
            yield utterance

if __name__ == "__main__":
    import doctest
    doctest.testmod()
