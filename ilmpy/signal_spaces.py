import warnings
import itertools
import string
from sympy.utilities.iterables import multiset_partitions as set_partitions

class _SoundSpace():
    """
    This is a private base class 
    """
    def __init__(self, noiserate = 0):
        self.noise_rate = noiserate

class SoundSpace (_SoundSpace):
    """
    >>> space      = SoundSpace(set('aeiou'))
    >>> space.sounds
    set(['a', 'i', 'e', 'u', 'o'])
    >>> space.schemata
    set(['a', 'e', 'i', 'u', '*', 'o'])
    >>> space.weights
    {'a': 1.0, 'e': 1.0, 'i': 1.0, '*': 0.0, 'o': 1.0, 'u': 1.0}
    """    
    def __init__(self, sounds, noiserate = 0):
        _SoundSpace.__init__(self, noiserate)
        self.sounds = sounds
        self.schemata = self.sounds | set('*')

        ## THESE WEIGHTS ARE FOR THE SMITH-KIRBY WEIGHTS FOR PRODUCTION AND RECEPTION
        weights = list([1.0] * len(self.sounds)) + list([0.0])
        self.weights  = dict(zip((list(sounds)+list('*')),weights))

    def generalize(self, sound):
        return ['*']

class TransformSoundSpace (_SoundSpace):    
    """
    >>> transform      = TransformSoundSpace('ae','AE')
    >>> transform.shortsounds
    'ae'
    >>> transform.longsounds
    'AE'
    >>> transform.sounds
    set(['a', 'A', 'e', 'E'])
    >>> transform.schemata
    set(['a', 'A', '#', 'e', '@', 'E'])
    >>> transform.weights
    {'a': 1.0, 'A': 1.0, '#': 0.0, 'e': 1.0, '@': 0.0, 'E': 1.0}
    """
    def __init__(self, shortsounds, longsounds, noiserate = 0):
        _SoundSpace.__init__(self, noiserate)
        if (len(shortsounds) != len(longsounds)):
            raise ValueError("Arguments to initialize TransformSoundSpace must be of equal length. You passed %s and %s" % (shortsounds,longsounds))
        if (len(shortsounds) > 12):
            raise ValueError("Only up to 12 transformable sound-pairs are supported. You passed %u" % (len(shortsounds)))
        self.shortsounds = shortsounds
        self.longsounds  = longsounds
        shortlong = shortsounds + longsounds
        longshort = longsounds + shortsounds
        self.translation_table = string.maketrans(shortlong,longshort)

        transform_wildcards = list("@#!+?$&%=<>.")[:len(shortsounds)]
        
        self._generalizations = dict(zip(list(shortlong),(transform_wildcards * 2))) ## limited to 12

        self.sounds   = set(shortsounds) | set (longsounds)
        self.schemata = self.sounds | set(transform_wildcards)
        
        ## THESE WEIGHTS ARE FOR THE SMITH-KIRBY WEIGHTS FOR PRODUCTION AND RECEPTION
        weights = list([1.0] * len(self.sounds)) + list([0.0] * len(transform_wildcards))
        self.weights  = dict(zip((list(shortlong)+transform_wildcards),weights))

    def generalize(self, sound):
        return [self._generalizations[sound]]
        

class _SignalSpace():
    """
    This is a private base class 
    """
    def __init__(self):
        self._signals = None
        self._schemata = None

class WordSignalSpace (_SignalSpace):
    """
    WordSignalSpace models natural utterances with a finite number of discrete sounds,
    a finite length, generalizable transformations on sounds, and anisotropic noise.

    For word models, nu defines the base noise rate and may be any number greater or equal to 0.
    The base noise rate is multiplied by dimension-specific noise rates given in the input argument
    This defines the per-symbol noise rate per transaction. 
    The probability of no change of a symbol is defined as (1 - nu).

    >>> signal_space = WordSignalSpace(nu = 0.1)
    >>> sounds1      = SoundSpace(set('bp'))
    >>> sounds2      = SoundSpace(set('aeiou'))
    >>> sounds3      = SoundSpace(set('dt'))
    
    >>> signal_space.add_component(sounds1)
    >>> signal_space.add_component(sounds2)
    >>> signal_space.add_component(sounds3)

    >>> set(signal_space.generalize('bad'))
    set(['b*d', 'b**', 'bad', '*a*', '*ad', '**d', 'ba*'])

    >>> list(signal_space.analyze('bad',2))
    [['**d', 'ba*'], ['*a*', 'b*d'], ['*ad', 'b**']]

    >>> list(signal_space.analyze('bad',3))
    [['*ad', 'b*d', 'ba*']]

    >>> sounds4      = TransformSoundSpace('ae','AE')
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
    """
    def __init__(self, nu = 0):
        _SignalSpace.__init__(self)
        self.nu = nu
        self.components = []
        self._weights = {}
        self.length = 0
        
    def add_component(self,component):
        self.components.append(component)
        self.length += 1
        symbols = []
        schemata = []
        keys     = []
        weights  = []
        for component in self.components:
            symbols.append(component.sounds)
            schemata.append(component.schemata)
            keys.append(component.weights.keys())
            weights.append(component.weights.values())
            
        self._signals = [''.join(s) for s in itertools.product(*symbols) ]
        self._schemata = [''.join(s) for s in itertools.product(*schemata) ]
        self._weights  = dict(zip(map(''.join,itertools.product(*keys)),map(sum,itertools.product(*weights))))
            
    def signals(self):
        return self._signals

    def schemata(self):
        return self._schemata

    def weights(self,schema):
        if (schema in self._weights):
            return (self._weights[schema] / self.length)
        else:
            None

    def analyze(self, signal, length):
        slist = list(signal)
        partitions = set_partitions(range(len(signal)),length)
        for partition in partitions:
            analysis = []
            for iset in partition:
                rlist = slist[:]
                for i in iset:
                    rlist[i] = self.components[i].generalize(rlist[i])[0]
                analysis.append(''.join(rlist))    
            yield analysis

    def generalize(self,signal):
        for i in range(len(signal)):
            for locs in itertools.combinations(range(len(signal)), i):
                sounds = [[char] for char in signal]
                for loc in locs:
                    original_sound = signal[loc]
                    sounds[loc] = self.components[loc].generalize(original_sound)
                for chars in itertools.product(*sounds):
                    schema = ''.join(chars)
                    yield schema 

if __name__ == "__main__":
    import doctest
    doctest.testmod()
