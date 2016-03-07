import warnings
import itertools
import string

class _SoundSpace():
    """
    This is a private base class 
    """
    def __init__(self, noiserate = 0):
        self.noise_rate = noiserate

class SoundSpace (_SoundSpace):
    """
    """    
    def __init__(self, sounds, noiserate = 0):
        _SoundSpace.__init__(self, noiserate)
        self.sounds = sounds | set('*')

    def generalize(self, sound):
        return ['*']

class TransformSoundSpace (_SoundSpace):    
    """
    """
    def __init__(self, shortsounds, longsounds, noiserate = 0):
        _SoundSpace.__init__(self, noiserate)
        if (len(shortsounds) != len(longsounds)):
            raise ValueError("Arguments to initialize TransformSoundSpace must be of equal length. You passed %s and %s" % (shortsounds,longsounds))
        self.shortsounds = shortsounds
        self.longsounds  = longsounds
        shortlong = shortsounds + longsounds
        longshort = longsounds + shortsounds
        self.translation_table = string.maketrans(shortlong,longshort)
        self.sounds = set(shortsounds) | set (longsounds)

    def generalize(self, sound):
        return [string.translate(sound,self.translation_table)] ## does this translate both directions?
        

class _SignalSpace():
    """
    This is a private base class 
    """
    def __init__(self):
        self.signals = None

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
    >>> set(signal_space.analyze('bad'))
    set(['b*d', 'b**', 'bad', '*a*', '*ad', '**d', 'ba*'])
    >>> sounds4      = TransformSoundSpace('ae','AE')
    >>> signal_space.add_component(sounds4)
    >>> set(signal_space.analyze('bada'))
    set(['*a*a', '*a*A', 'bada', '***a', '**da', '*ada', 'b**a', 'ba*A', '*adA', 'badA', 'ba*a', '**dA', 'b**A', 'b*dA', 'b*da'])
    >>> set(signal_space.analyze('badA'))
    set(['*a*A', '*a*a', 'badA', '***A', '**dA', '*adA', 'b**A', 'ba*a', '**da', '*ada', 'bada', 'ba*A', 'b**a', 'b*da', 'b*dA'])
    
    """
    def __init__(self, nu = 0):
        _SignalSpace.__init__(self)
        self.nu = nu
        self.components = []
        self.component_added = False
        
    def add_component(self,component):
        self.components.append(component)
        self.component_added = True

    def generalize(self,register,sound):
        return self.components[register].generalize(sound)

    def analyze(self,signal):
        for i in range(len(signal)):
            for locs in itertools.combinations(range(len(signal)), i):
                sounds = [[char] for char in signal]
                for loc in locs:
                    original_sound = signal[loc]
                    sounds[loc] = self.components[loc].generalize(original_sound)
                for chars in itertools.product(*sounds):
                    yield ''.join(chars)

    def signals(self):
        if self.signals is None or self.component_added:
            symbols = []
            for component in self.components:
                symbols.append(component.sounds)
                self.signals = itertools.product(symbols)
            self.component_added = False
        return self.signals
        

if __name__ == "__main__":
    import doctest
    doctest.testmod()
