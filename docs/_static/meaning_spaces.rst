meaning_spaces.py
========================================

CLASS _MeaningComponent
--------------------------


Purpose
^^^^^^^^^^^^^^^^^


Usage
^^^^^^^^^^^^^^^^^

* Using a terminal, navigate to ILMpy/ilmpy or the folder that contains argument_parser.py 
* Type ``python`` into terminal to start python interpreter
* Type the following into python terminal, the prompt should show >>>

	
Code Walkthrough
^^^^^^^^^^^^^^^^^

Learner Class does what::

	class _MeaningComponent():
	    """
	    This is a private base class 
	    """
	    def __init__(self, size):
	        # check value
	        self.size = size
	        self._meanings = set([str(i) for i in list(range(size))]) # meanings are vectors of integers and graph nodes
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


CLASS OrderedMeaningComponent
--------------------------------


Purpose
^^^^^^^^^^^^^^^^^


Usage
^^^^^^^^^^^^^^^^^

* Using a terminal, navigate to ILMpy/ilmpy or the folder that contains argument_parser.py 
* Type ``python`` into terminal to start python interpreter
* Type the following into python terminal, the prompt should show >>>

OrderedMeaningComponent::
	>>> omc = OrderedMeaningComponent(5)
	>>> omc.generalize(4)
	['*']
	>>> omc.meanings()
	set(['1', '0', '3', '2', '4'])
	>>> omc.schemata()
	set(['1', '0', '3', '2', '4', '*'])
	>>> omc.weights()
	{'*': 0.0, '1': 1.0, '0': 1.0, '3': 1.0, '2': 1.0, '4': 1.0}

Code Walkthrough
^^^^^^^^^^^^^^^^^

Code::

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
	        if not str(meaning) in self._meanings:
	            raise ValueError('unknown meaning component {}'.format(meaning))
	        return  ['*']
	


CLASS UnorderedMeaningComponent
--------------------------------


Purpose
^^^^^^^^^^^^^^^^^


Usage
^^^^^^^^^^^^^^^^^

* Using a terminal, navigate to ILMpy/ilmpy or the folder that contains argument_parser.py 
* Type ``python`` into terminal to start python interpreter
* Type the following into python terminal, the prompt should show >>>

OrderedMeaningComponent::
	>>> omc = OrderedMeaningComponent(5)
	>>> omc.generalize(4)
	['*']
	>>> omc.meanings()
	set(['1', '0', '3', '2', '4'])
	>>> omc.schemata()
	set(['1', '0', '3', '2', '4', '*'])
	>>> omc.weights()
	{'*': 0.0, '1': 1.0, '0': 1.0, '3': 1.0, '2': 1.0, '4': 1.0}

Code Walkthrough
^^^^^^^^^^^^^^^^^

Code::

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
	    set(['1', '0', '3', '2', '4'])
	    >>> umc.weights()
	    {'1': 1.0, '0': 1.0, '3': 1.0, '2': 1.0, '4': 1.0}
	    """
	    def __init__(self, size):
	        _MeaningComponent.__init__(self,size)
	        self._schemata = self._meanings.copy()
	        weights = list([1.0] * len(self._meanings))
	        self._weights  = dict(zip((list(self._meanings)),weights))        
	
	    def generalize(self, meaning):
	        return [meaning]; # the generalization identity 





CLASS _MeaningSpace
--------------------------------


Purpose
^^^^^^^^^^^^^^^^^


Usage
^^^^^^^^^^^^^^^^^

* Using a terminal, navigate to ILMpy/ilmpy or the folder that contains argument_parser.py 
* Type ``python`` into terminal to start python interpreter
* Type the following into python terminal, the prompt should show >>>

OrderedMeaningComponent::

	>>> omc = OrderedMeaningComponent(5)
	>>> omc.generalize(4)
	['*']
	>>> omc.meanings()
	set(['1', '0', '3', '2', '4'])
	>>> omc.schemata()
	set(['1', '0', '3', '2', '4', '*'])
	>>> omc.weights()
	{'*': 0.0, '1': 1.0, '0': 1.0, '3': 1.0, '2': 1.0, '4': 1.0}

Code Walkthrough
^^^^^^^^^^^^^^^^^

Code::

	class _MeaningSpace():
    """
    This is a private base class 
    """
    def __init__(self):
        self._meanings = None
        self._schemata = None
        self._weights  = None


CLASS: CombinatorialMeaningSpace
--------------------------------


Purpose
^^^^^^^^^^^^^^^^^


Usage
^^^^^^^^^^^^^^^^^

* Using a terminal, navigate to ILMpy/ilmpy or the folder that contains argument_parser.py 
* Type ``python`` into terminal to start python interpreter
* Type the following into python terminal, the prompt should show >>>

CombinatorialMeaningSpace::

	>>> meaning_space = CombinatorialMeaningSpace()
	>>> meanings1    = OrderedMeaningComponent(3)
	>>> meanings2    = UnorderedMeaningComponent(2)
	>>> meanings3    = OrderedMeaningComponent(2)
	
	>>> meaning_space.add_component(meanings1)
	>>> meaning_space.add_component(meanings2)
	>>> meaning_space.add_component(meanings3)
	
	>>> set(meaning_space.generalize('1.1.1'))
	set(['1.1.1', '*.1.*', '*.1.1', '1.1.*'])
	
	>>> list(meaning_space.analyze('1.1.1',2))
	[['*.1.1', '1.1.*'], ['*.1.*', '1.1.1'], ['*.1.1', '1.1.*']]
	
	>>> list(meaning_space.analyze('1.1.1',3))
	[['*.1.1', '1.1.1', '1.1.*']]
	
	>>> meaning_space.meanings()
	['1.1.1', '1.1.0', '1.0.1', '1.0.0', '0.1.1', '0.1.0', '0.0.1', '0.0.0', '2.1.1', '2.1.0', '2.0.1', '2.0.0']
	
	>>> meaning_space.schemata()
	['1.1.1', '1.1.0', '1.1.*', '1.0.1', '1.0.0', '1.0.*', '0.1.1', '0.1.0', '0.1.*', '0.0.1', '0.0.0', '0.0.*', '2.1.1', '2.1.0', 	'2.1.*', '2.0.1', '2.0.0', '2.0.*', '*.1.1', '*.1.0', '*.1.*', '*.0.1', '*.0.0', '*.0.*']
	
	>>> meaning_space.sample(10)
	
	>>> meaning_space.hamming('100','011')
	1.0
	
	>>> meanings4 = OrderedMeaningComponent(12)
	>>> meaning_space.add_component(meanings4)
	>>> set(meaning_space.generalize('1.1.1.14'))
	ValueError

Code Walkthrough
^^^^^^^^^^^^^^^^^

Code::

	class CombinatorialMeaningSpace (_MeaningSpace):

    def __init__(self):
        _MeaningSpace.__init__(self)
        self._components = []
        self._weights = {}
        self._hamming = defaultdict(dict)
        self.length = 0

    def add_component(self,component):
        ## self.components.append(component)
        ## self.length += 1
        ## meanings = []
        ## schemata = []
        ## keys     = []
        ## weights  = []
        ## for component in self.components:
        ##     meanings.append(component.meanings())
        ##     schemata.append(component.schemata())
        ##     keys.append(component.weights().keys())
        ##     weights.append(component.weights().values())
            
        ## self._meanings = [''.join(s) for s in itertools.product(*meanings) ]
        ## self._schemata = [''.join(s) for s in itertools.product(*schemata) ]
        ## self._weights  = dict(zip(map(''.join,itertools.product(*keys)),map(sum,itertools.product(*weights))))

        if (self.length == 0):
            self._meanings      = [ '.'.join(m) for m in itertools.product(component.meanings()) ]
            self._schemata      = [ '.'.join(s) for s in itertools.product(component.schemata()) ]
            self._weightkeys    = [ '.'.join(k) for k in itertools.product(component.weights().keys()) ]
            self._weightvalues  = [     sum(v) for v in itertools.product(component.weights().values()) ]
            self._weights       = dict(zip(self._weightkeys,self._weightvalues))
        else:
            self._meanings      = [ '.'.join(m) for m in itertools.product(self._meanings,component.meanings()) ]
            self._schemata      = [ '.'.join(s) for s in itertools.product(self._schemata,component.schemata()) ]
            self._weightkeys    = [ '.'.join(k) for k in itertools.product(self._weightkeys,component.weights().keys()) ]
            self._weightvalues  = [     sum(v) for v in itertools.product(self._weightvalues,component.weights().values()) ]
            self._weights       = dict(zip(self._weightkeys,self._weightvalues))

        self.length += 1
        self._components.append(component)

        ## remove the all-general component from schemata
        

    def components(self,i):
         return self._components[i]

    def meanings(self):
        return self._meanings

    def schemata(self):
        return self._schemata

    def weights(self,schema):
        if (schema in self._weights):
            return (self._weights[schema] / self.length)
        else:
            None

    def hamming(self,mean1,mean2):
        assert len(mean1.split('.')) == len(mean2.split('.'))
        if (mean1 == mean2):
            return 0
        elif mean1 in self._hamming and mean2 in self._hamming[mean1]:
            return self._hamming[mean1][mean2]
        else:
            marray1 = numpy.array(mean1.split('.'))
            marray2 = numpy.array(mean2.split('.'))
            hd = numpy.count_nonzero(marray1!=marray2)
            self._hamming[mean1][mean2] = self._hamming[mean2][mean1] = (hd/self.length)
            return self._hamming[mean1][mean2]

    def analyze(self, meaning, length):
        ## import pdb
        ## pdb.set_trace()
        mlist = meaning.split('.')
        partitions = set_partitions(range(len(mlist)),length)
        for partition in partitions:
            analysis = []
            for iset in partition:
                rlist = mlist[:]
                for i in iset:
                    rlist[i] = self.components(i).generalize(rlist[i])[0]
                analysis.append('.'.join(rlist))    
            yield analysis

    def generalize(self,meaning):
        #import pdb
        #pdb.set_trace()
        mlist = meaning.split('.')
        for i in range(len(mlist)):
            for locs in itertools.combinations(range(len(mlist)), i):
                meanings = [[component] for component in mlist]
                for loc in locs:
                    original_meaning = mlist[loc]
                    meanings[loc] = self.components(loc).generalize(original_meaning)
                for components in itertools.product(*meanings):
                    schema = '.'.join(components)
                    yield schema 
    
    def sample(self,number):
        if (number < 0 or (number != floor(number))):
            raise ValueError("Parameter number must be an integer >= 0. You passed %f" % (number))
        return sample(self._meanings,number) # samples without replacement











