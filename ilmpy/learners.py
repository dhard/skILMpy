import warnings
import pandas
import numpy
import pdb
import signal_spaces,meaning_spaces
import random
import itertools

class _Learner ():
    """
    This is a private base class 
    """
    def __init__(self, meaning_space,signal_space):
        self.meaning_space = meaning_space
        self.signal_space = signal_space

    def learn (self, data):
        """
        Learn associations from a list of signal-meaning pairs
        """
        pass

    def hear (self, signal):
        """
        Returns the meaning for a signal
        """
        if (signal not in self.signal_space.signals() ):
            raise ValueError("Signal unrecognized. You passed %s" % (signal))
    
    def think (self, number):
        """
        Returns a list of a specified number of random meanings
        """
        if (number < 0 or (number != floor(number))):
            raise ValueError("Parameter number must be an integer >= 0. You passed %f" % (number))


class AssociationMatrixLearner (_Learner):
    """
    This class implements the original Smith-Kirby ILM

    >>> signal_space = signal_spaces.WordSignalSpace()
    >>> sounds1      = signal_spaces.SignalComponent(set('bp'))
    >>> sounds3      = signal_spaces.SignalComponent(set('dt'))
    
    >>> signal_space.add_component(sounds1)
    >>> signal_space.add_component(sounds3)

    >>> meaning_space = meaning_spaces.CombinatorialMeaningSpace()
    >>> meanings1     = meaning_spaces.OrderedMeaningComponent(2)
    >>> meanings3     = meaning_spaces.OrderedMeaningComponent(2)

    >>> meaning_space.add_component(meanings1)
    >>> meaning_space.add_component(meanings3)

    >>> child = AssociationMatrixLearner(meaning_space,signal_space)
    >>> child.learn([['00','bd',1.0]])
    >>> child.speak('00')
    'bd'

    >>> signal_space = signal_spaces.WordSignalSpace()
    >>> sounds1      = signal_spaces.SignalComponent(set('bp'))
    >>> sounds2      = signal_spaces.TransformSignalComponent('aeiou','AEIOU',noiserate=0.1)
    >>> sounds3      = signal_spaces.SignalComponent(set('dt'))
    
    >>> signal_space.add_component(sounds1)
    >>> signal_space.add_component(sounds2)
    >>> signal_space.add_component(sounds3)

    >>> meaning_space = meaning_spaces.CombinatorialMeaningSpace()
    >>> meanings1     = meaning_spaces.OrderedMeaningComponent(2)
    >>> meanings2     = meaning_spaces.OrderedMeaningComponent(5)
    >>> meanings3     = meaning_spaces.OrderedMeaningComponent(2)

    >>> meaning_space.add_component(meanings1)
    >>> meaning_space.add_component(meanings2)
    >>> meaning_space.add_component(meanings3)

    >>> founder = AssociationMatrixLearner(meaning_space,signal_space, alpha=1, beta=0, gamma=-1, delta=-1, seed=42, amplitude = 0.25)
    >>> lessons = founder.teach(20)
    >>> lessons
    [[['001', 'pEd', 0.9], ['001', 'ped', 0.1]], [['111', 'bUd', 0.9], ['111', 'bud', 0.1]], [['131', 'pId', 0.9], ['131', 'pid', 0.1]], [['100', 'bad', 0.9], ['100', 'bAd', 0.1]], [['010', 'pEd', 0.9], ['010', 'ped', 0.1]], [['011', 'bUd', 0.9], ['011', 'bud', 0.1]], [['040', 'pEd', 0.9], ['040', 'ped', 0.1]], [['110', 'bet', 0.9], ['110', 'bEt', 0.1]], [['130', 'pAd', 0.9], ['130', 'pad', 0.1]], [['041', 'ped', 0.9], ['041', 'pEd', 0.1]], [['101', 'pAd', 0.9], ['101', 'pad', 0.1]], [['020', 'pud', 0.9], ['020', 'pUd', 0.1]], [['031', 'pAd', 0.9], ['031', 'pad', 0.1]], [['000', 'bad', 0.9], ['000', 'bAd', 0.1]], [['021', 'pEd', 0.9], ['021', 'ped', 0.1]], [['140', 'bUd', 0.9], ['140', 'bud', 0.1]], [['120', 'pid', 0.9], ['120', 'pId', 0.1]], [['121', 'bUd', 0.9], ['121', 'bud', 0.1]], [['141', 'bEt', 0.9], ['141', 'bet', 0.1]], [['030', 'bad', 0.9], ['030', 'bAd', 0.1]]]
    >>> child = founder.spawn()
    >>> child.learn(lessons)
    >>> child.speak('001')
    'pod'


    """
    def __init__(self,meaning_space, signal_space, alpha=1, beta=-1, gamma=-1, delta=0, observables=None, seed=None, amplitude=None):
        _Learner.__init__(self, meaning_space, signal_space)
        if (seed):
            numpy.random.seed(seed)
            random.seed(seed)
        if (amplitude):
            values = (2 * amplitude) * numpy.random.random_sample((len(meaning_space.schemata()), len(signal_space.schemata()))) - amplitude
        else:
            values = 0
        self.matrix = pandas.DataFrame(values,index=meaning_space.schemata(), columns=signal_space.schemata())  
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.observables = observables

    def spawn(self):
        child = AssociationMatrixLearner(self.meaning_space,self.signal_space,alpha=self.alpha,beta=self.beta,gamma=self.gamma,delta=self.delta, observables=self.observables)
        return child

    def score_meaning(self,meaning_schema,signal_schema):
        weight  = self.signal_space.weights(signal_schema)
        strength = self.matrix.loc[meaning_schema,signal_schema]
        return weight * strength

    def score_signal(self,meaning_schema,signal_schema):
        weight = self.meaning_space.weights(meaning_schema)
        strength = self.matrix.loc[meaning_schema,signal_schema]
        return weight * strength
        
    def learn(self,data):
        """
        Learn associations from a list of signal-meaning pairs
        """
        #        pdb.set_trace()
        for datum in data:
            meaning = datum[0]
            signal = datum[1]
            freq_weight = datum[2]

            self.matrix                                   += (self.delta * freq_weight)
            for signal_schema in self.signal_space.generalize(signal):
                self.matrix.loc[:,signal_schema]          += ((self.gamma - self.delta) * freq_weight)

            for meaning_schema in self.meaning_space.generalize(meaning):
                self.matrix.loc[meaning_schema,:]         += ((self.beta - self.delta)  * freq_weight)

            for signal_schema in self.signal_space.generalize(signal):
                for meaning_schema in self.meaning_space.generalize(meaning):
                    self.matrix.loc[meaning_schema,signal_schema] += ((self.alpha - self.beta - self.gamma + self.delta) * freq_weight)
 

    def hear (self, signal):
        """
        Return the optimal meaning for a signal
        """
        meanings = self.meaning_space.meanings()
        winners = []
        maxscore = None
        for analysis_size in xrange(2,(len(signal)+1)):
            for signal_analysis in self.signal_space.analyze(signal,analysis_size):
                for meaning in meanings:
                    for meaning_analysis in self.meaning_space.analyze(meaning,analysis_size):
                        for permutation in itertools.permutations(meaning_analysis):
                            pairs = zip(signal_analysis, permutation)
                            score = 0
                            for signal_schema,meaning_schema in pairs:
                                score += self.score_meaning(meaning_schema,signal_schema)
                            if (score > maxscore):
                                maxscore = score
                                winners = [meaning]
                            elif (score == maxscore):
                                winners.append(meaning)

        if (len(winners) == 1):
            return winners[0]
        else:
            return random.choice(winners)        

    def speak (self, meaning):
        """
        Produce a signal corresponding to a meaning
        """
        signals = self.signal_space.signals()
        winners = []
        maxscore = None
        for analysis_size in xrange(2,(len(meaning)+1)):
            for meaning_analysis in self.meaning_space.analyze(meaning,analysis_size):
                for signal in signals:
                    for signal_analysis in self.signal_space.analyze(signal,analysis_size):
                        for permutation in itertools.permutations(signal_analysis):
                            pairs = zip(permutation,meaning_analysis)
                            score = 0
                            for signal_schema,meaning_schema in pairs:
                                score += self.score_signal(meaning_schema,signal_schema)
                            if (score > maxscore):
                                maxscore = score
                                winners = [signal]
                            elif (score == maxscore):
                                winners.append(signal)                          
        if (len(winners) == 1):
            return winners[0]
        else:
            return random.choice(winners) 

    def think(self, number):
        """
        Returns a list of a specified number of random meanings
        """
        return self.meaning_space.sample(number)
                    
    def teach(self,number):
        """
        Returns a specified number of list of pairs of random meanings and best signals learned for them.
        Provide each meaning-signal pair with a frequency weight 
        """
        thoughts   = self.think(number)
        frequency  = 1.0
        lessons = [ [thought, self.speak(thought), frequency ] for thought in thoughts ]
        if (self.signal_space.noisy):
            distortions = []
            for thought,utterance,freq in lessons:
                distortions.append([ [thought, distortion, frequency] for distortion, frequency in self.signal_space.distort(utterance) ])
            return distortions
        else:
            return lessons

    def print_initial_observables(self):
        if (self.observables.show_initial_parameters):
            print '# Delta:{} Epsilon:{}'.format(self.delta,self.epsilon)
        if (self.observables.show_matrix_parameters):
            precision = self.observables.print_precision
            mm = self.codons.get_mutation_matrix()
            print "# Mutation Matrix:\n",mm
            dm = self.aas.get_distance_matrix()
            print "# Distance matrix:\n",dm.round(precision)
            fm = site_types.get_fitness_matrix()
            print "# Fitness matrix:\n",fm.round(precision)
            msm = self.get_mutation_selection_matrix(0)
            print "# Iteration Matrix for site-type 0:\n",msm.round(precision)


    def print_observables_header(self):
        if self.observables.show_code_evolution_statistics or self.observables.show_all:
            print '# RBE = Reassignments Before Explicit'
            print '# RAE = Reassignments After Explicit'
            print '# NAA = Number encoded Amino Acids'
            print '# NER = Normalized Encoded Range (Ardell and Sella, 2001)'
        if self.observables.show_fitness_statistics or self.observables.show_all:
            print '# NFCM = Number Fitter Code Mutants'
            print '# MMF  = Maximum Mutant Fitness'
            print '# GR   = Growth Rate'
            print '# GRFL = Growth Rate from Lambda'
            print '#'
		
        precision = self.observables.print_precision
        width = precision + 8
        print '#{:>{width}s}'.format('STEPS',width=(width-1)),
        if self.observables.show_code_evolution_statistics or self.observables.show_all:
            obs = ['RBE','RAE','NAA','NER']
            print ('{:>{width}s}'*(len(obs))).format(*obs,width=width),
        if self.observables.show_fitness_statistics or self.observables.show_all:
            obs = ['NFCM','MMF','GR','GRFL']
            print ('{:>{width}s}'*(len(obs))).format(*obs,width=width),
        print	      


	def print_observables(self):
            if self.observables.show_codes or self.observables.show_all:
                print '{}'.format(self.code)

            precision = self.observables.print_precision
            width = precision + 8

            print '{:{width}d}'.format(self.code.num_mutations,width=width),

            #if self.observables.show_codes_single_line:
            #	print '{}\t'.format(self.code.as_string),

            if self.observables.show_code_evolution_statistics or self.observables.show_all:
                obs = [self.code.num_reassignments_before_explicit(),self.code.num_reassignments_after_explicit(),self.code.num_encoded_amino_acids()]
                print ('{:>{width}d}'*(len(obs))).format(*obs,width=width),
                ner = self.code.normalized_encoded_range()
                if ner.__class__.__name__ == 'str':
                    type = 's'
                else:
                    type = 'f'
                print '{:>{width}.{precision}{type}}'.format(self.code.normalized_encoded_range(),width=width-1,precision=precision,type=type),
            if self.observables.show_fitness_statistics or self.observables.show_all:
                print '{:>{width}d}'.format(self.num_fitter_code_mutants,width=width),
                obs = [self.max_mutant_fitness,self.growth_rate(),self.growth_rate_from_lambda()]
                print ('{:>{width}.{precision}f}'*(len(obs))).format(*obs,width=width,precision=precision),
            if self.observables.show_messages:
                print self.messages().round(precision),
            print "\n"


if __name__ == "__main__":
    import doctest
    doctest.testmod()
