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
        self.shortsounds = shortsounds
        self.longsounds  = longsounds
        self.translation_table = string.maketrans(shortsounds,longsounds)
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

    """
    def __init__(self, nu = 0):
        _SignalSpace.__init__(self)
        self.nu = nu
        self.components = []
        
    def add_component(self,component):
        self.components.append(component)

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
        if self.signals is None:
            symbols = []
            for component in self.components:
                symbols.append(component.sounds)
                self.signals = itertools.product(symbols)
        return self.signals
        

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
