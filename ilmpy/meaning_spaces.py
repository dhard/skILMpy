import warnings
import itertools
import string
import ilmpy

class _MeaningComponent():
    """
    This is a private base class 
    """
    def __init__(self, size):
        self.size = size
        self.meanings = xrange(size)

    def generalize(self, meaning):
        return  ['*']

class OrderedMeaningComponent (_MeaningComponent):
    """
    >>> omc = ilmpy.meaning_spaces.OrderedMeaningComponent(5)
    """    
    def __init__(self, size):
        _MeaningComponent.__init__(self, size)
        
class UnorderedMeaningComponent (_MeaningComponent):    
    """
    >>> umc = ilmpy.meaning_spaces.OrderedMeaningComponent(5)

    """
    def __init__(self, size):
        _MeaningComponent.__init__(self, size)
    
    def generalize(self, meaning):
        return meaning;

class _MeaningSpace():
    """
    This is a private base class 
    """
    def __init__(self):
        self.meanings = None

class CombinatorialMeaningSpace (_MeaningSpace):
    """
    """
    def __init__(self):
        _MeaningSpace.__init__(self)
        self.components = []
        
    def add_component(self,component):
        self.components.append(component)

    def generalize(self,dimension,meaning):
        return self.components[dimension].generalize(meaning)

    def analyze(self,meaning):
        for i in range(len(meaning)):
            for locs in itertools.combinations(range(len(meaning)), i):
                components = [[m] for m in meaning]
                for loc in locs:
                    original_component = meaning[loc]
                    meanings[loc] = self.components[loc].generalize(original_component)
                for chars in itertools.product(*meanings):
                    yield ''.join(chars)

    def meanings(self):
        if self.meanings is None:
            meanings = []
            for component in self.components:
                meanings.append(component.meanings)
                self.meanings = itertools.product(meanings)
        return self.meanings


if __name__ == "__main__":
    import doctest
    doctest.testmod()
   
