from __future__ import division
from __future__ import print_function 
import warnings
import pandas
import numpy
import pdb
import ilmpy.signal_spaces as signal_spaces
import ilmpy.meaning_spaces as meaning_spaces
import random
import copy
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
    [['001', 'pEd', 0.9], ['001', 'ped', 0.1], ['111', 'bUd', 0.9], ['111', 'bud', 0.1], ['131', 'pId', 0.9], ['131', 'pid', 0.1], ['100', 'bad', 0.9], ['100', 'bAd', 0.1], ['010', 'pEd', 0.9], ['010', 'ped', 0.1], ['011', 'bUd', 0.9], ['011', 'bud', 0.1], ['040', 'pEd', 0.9], ['040', 'ped', 0.1], ['110', 'bet', 0.9], ['110', 'bEt', 0.1], ['130', 'pAd', 0.9], ['130', 'pad', 0.1], ['041', 'ped', 0.9], ['041', 'pEd', 0.1], ['101', 'pAd', 0.9], ['101', 'pad', 0.1], ['020', 'pud', 0.9], ['020', 'pUd', 0.1], ['031', 'pAd', 0.9], ['031', 'pad', 0.1], ['000', 'bad', 0.9], ['000', 'bAd', 0.1], ['021', 'pEd', 0.9], ['021', 'ped', 0.1], ['140', 'bUd', 0.9], ['140', 'bud', 0.1], ['120', 'pid', 0.9], ['120', 'pId', 0.1], ['121', 'bUd', 0.9], ['121', 'bud', 0.1], ['141', 'bEt', 0.9], ['141', 'bet', 0.1], ['030', 'bad', 0.9], ['030', 'bAd', 0.1]]
    >>> child = founder.spawn()
    >>> child.learn(lessons)
    >>> child.speak('001')
    'pEd'

    """
    def __init__(self,meaning_space, signal_space, alpha=1, beta=-1, gamma=-1, delta=0, observables=None, amplitude=None):
        _Learner.__init__(self, meaning_space, signal_space)
        #pdb.set_trace()
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
        self._matrix_updated = False
        self._speak = {}
        self._hear = {}

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
        #pdb.set_trace()
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

        self._matrix_updated = True

    def hear (self, signal, pick = True):
        """
        Return the optimal meaning for a signal
        """
        if self._matrix_updated or not signal in self._hear:
            meanings = self.meaning_space.meanings()
            winners = []
            maxscore = None
            for analysis_size in range(2,(len(signal)+1)):
                for signal_analysis in self.signal_space.analyze(signal,analysis_size):
                    for meaning in meanings:
                        for meaning_analysis in self.meaning_space.analyze(meaning,analysis_size):
                            for permutation in itertools.permutations(meaning_analysis):
                                pairs = zip(signal_analysis, permutation)
                                score = 0
                                for signal_schema,meaning_schema in pairs:
                                    score += self.score_meaning(meaning_schema,signal_schema)
                                if (not maxscore or score > maxscore):
                                    maxscore = score
                                    winners = [meaning]
                                elif (score == maxscore):
                                    winners.append(meaning)
            if pick:
                if (len(winners) == 1):
                    winner = winners[0]
                else:
                    winner = random.choice(winners) 
            else:
                winner = winners
                
            self._matrix_updated = False
            self._hear[signal] = winners
            return winner
        else:
            if pick:
                if (len(self._hear[signal]) == 1):
                    return self._hear[signal][0]
                else:
                    return random.choice(self._hear[signal])         
            else:
                return self._hear[signal]

    def speak (self, meaning, pick = True):
        """
        Produce a signal corresponding to a meaning
        """
        if self._matrix_updated or not meaning in self._speak:
            signals = self.signal_space.signals()
            winners = []
            maxscore = None   
            for analysis_size in range(2,(len(meaning)+1)):
                for meaning_analysis in self.meaning_space.analyze(meaning,analysis_size):
                    for signal in signals:
                        for signal_analysis in self.signal_space.analyze(signal,analysis_size):
                            for permutation in itertools.permutations(signal_analysis):
                                pairs = zip(permutation,meaning_analysis)
                                score = 0
                                for signal_schema,meaning_schema in pairs:
                                    score += self.score_signal(meaning_schema,signal_schema)
                            
                              
                                if (not maxscore or score > maxscore):
                                    maxscore = score
                                    winners = [signal]
                                elif (score == maxscore and signal not in winners):
                                    winners.append(signal)                          
            if pick:              
                if (len(winners) == 1):
                    winner = winners[0]
                else:
            
                    winner = random.choice(winners) 
                    
            else:
                winner = winners

            self._matrix_updated = False
            self._speak[meaning] = winners
            return winner
        else:
            if pick:
                if (len(self._speak[meaning]) == 1):
                    return self._speak[meaning][0]
                else:
                   
                    return random.choice(self._speak[meaning]) 
            else:
                return self._speak[meaning]

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
                distortions.extend([[thought, distortion, frequency] for distortion, frequency in self.signal_space.distort(utterance) ])
            if self.observables and self.observables.show_lessons:
                print("lessons: ",distortions)
            return distortions
        else:
            if self.observables and self.observables.show_lessons:
                print("lessons: ",lessons)
            return lessons

    def vocabulary(self):
        """
        Returns all meanings and optimal signals learned for them.
        """        
        thoughts = self.meaning_space.meanings()
        vocabulary = [ [thought, self.speak(thought, pick=False) ] for thought in thoughts ]
        return vocabulary

    def compute_compositionality(self):
        """
        Computes a compositionality measure related to the one introduced in Sella Ardell (2001) DIMACS
        """
        #pdb.set_trace()
        compositionality = 0
        comparisons = 0
        meanings = self.meaning_space.meanings()
        for meaning1,meaning2 in itertools.combinations(meanings, 2):
            mdist = self.meaning_space.hamming(meaning1,meaning2)
            signals1 = self.speak(meaning1, pick=False)
            signals2 = self.speak(meaning2, pick=False)
            for signal1 in signals1:
                for signal2 in signals2:
                    sdist = self.signal_space.hamming(signal1,signal2)
                    compositionality += ((mdist * sdist) / (len(signals1) * len(signals2)))
                    comparisons += 1
        #pdb.set_trace()       
        return (compositionality/comparisons)

    def compute_accuracy(self):
        """
        Computes the Communicative Accuracy of self e.g. Brighton et al (2005) eq.A.1 
        """
        #pdb.set_trace()
        accuracy = 0
        meanings = self.meaning_space.meanings()
        for meaning in meanings:
            utterances = self.speak(meaning, pick=False)
            for utterance in utterances:
                understandings = self.hear(utterance, pick=False)
                if meaning in understandings:
                    accuracy += (1/len(utterances)) * (1/len(understandings))
        #pdb.set_trace()
        return (accuracy/len(meanings))

    def compute_load(self):
        """
        Calculates the functional load by signal position, the hamming distance of meanings induced by changes in each position
        """
        #pdb.set_trace()
        load = [ 0 for _ in range(self.signal_space.length) ]
        meanings = self.meaning_space.meanings()
        for position in range(self.signal_space.length):
            comparisons = 0
            for meaning in meanings:
                utterances = self.speak(meaning, pick=False)
                for utterance in utterances:
                    neighbors = self.signal_space.compute_neighbors(utterance,position)
                    for neighbor in neighbors:
                        understandings = self.hear(neighbor, pick=False)
                        for understanding in understandings:
                            mdist = self.meaning_space.hamming(meaning,understanding)
                            load[position] += (mdist /  self.meaning_space.length)
                            comparisons    += 1
            load[position] /= comparisons
        pdb.set_trace()
        return load

    def compute_entropy(self):
        """
        Calculates the symbol Shannon entropy of the vocabulary by signal position
        """
        #pdb.set_trace()
        vocab = self.vocabulary()
        for position in range(self.signal_space.length):
            comparisons = 0
            for meaning in meanings:
                utterances = self.speak(meaning, pick=False)
                for utterance in utterances:
                    neighbors = self.signal_space.compute_neighbors(utterance,position)
                    for neighbor in neighbors:
                        understandings = self.hear(neighbor, pick=False)
                        for understanding in understandings:
                            mdist = self.meaning_space.hamming(meaning,understanding)
                            load[position] += (mdist /  self.meaning_space.length)
                            comparisons    += 1
            load[position] /= comparisons
        #pdb.set_trace()
        return load

    def print_parameters(self):
        params = {'alpha':self.alpha, 'beta':self.beta, 'gamma':self.gamma, 'delta':self.delta}#, 'interactions": }
        precision = self.observables.print_precision
        width = precision + 8
        print("# params: ",'alpha: {alpha}  beta: {beta} gamma: {gamma} delta: {delta}'.format(**params))


    def print_observables_header(self):
        obs = []
        precision = self.observables.print_precision
        width = precision + 8
        if self.observables.show_compositionality or self.observables.show_stats:
            print('# COM = Compositionality')
            obs.append('COM')
        if self.observables.show_accuracy or self.observables.show_stats:
            print('# ACC = Communicative Self-Accuracy')
            obs.append('ACC')
        if self.observables.show_load or self.observables.show_stats:            
            print('# FLD = Functional Load by Signal Position, One for Each')
            obs.append('FLD')
        if obs:
            print(('{:>{width}s}'*(len(obs))).format(*obs,width=width))


    def print_observables(self):
        if self.observables.show_matrices:
            print(self.matrix)

        obs = []
        precision = self.observables.print_precision
        width = precision + 8
        if self.observables.show_compositionality or self.observables.show_stats:
            obs.append(self.compute_compositionality())
        if self.observables.show_accuracy or self.observables.show_stats:
            obs.append(self.compute_accuracy())
        if self.observables.show_load or self.observables.show_stats:            
            obs.extend(self.compute_load())

        if obs:
            print("stats: ",('{:>{width}f}'*(len(obs))).format(*obs,width=width))

        if self.observables.show_vocabulary:
            print("vocabulary: ", self.vocabulary())

if __name__ == "__main__":
    import doctest
    doctest.testmod()
