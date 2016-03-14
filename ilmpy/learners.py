import warnings
import pandas
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

    >>> signal_space = signal_spaces.WordSignalSpace(nu = 0.1)
    >>> sounds1      = signal_spaces.SoundSpace(set('bp'))
    >>> sounds3      = signal_spaces.SoundSpace(set('dt'))
    
    >>> signal_space.add_component(sounds1)
    >>> signal_space.add_component(sounds3)

    >>> meaning_space = meaning_spaces.CombinatorialMeaningSpace()
    >>> meanings1     = meaning_spaces.OrderedMeaningComponent(2)
    >>> meanings3     = meaning_spaces.OrderedMeaningComponent(2)

    >>> meaning_space.add_component(meanings1)
    >>> meaning_space.add_component(meanings3)

    >>> child = AssociationMatrixLearner(meaning_space,signal_space)
    >>> child.learn([['00','bd']])
    >>> child.speak('00')
    'bd'

    >>> signal_space = signal_spaces.WordSignalSpace(nu = 0.1)
    >>> sounds1      = signal_spaces.SoundSpace(set('bp'))
    >>> sounds2      = signal_spaces.SoundSpace(set('aeiou'))
    >>> sounds3      = signal_spaces.SoundSpace(set('dt'))
    
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

    >>> child = AssociationMatrixLearner(meaning_space,signal_space)
    >>> child.learn([['020','bad']])


    """
    def __init__(self,meaning_space, signal_space, alpha=1, beta=-1, gamma=-1, delta=0):
        _Learner.__init__(self, meaning_space, signal_space)
        self.matrix = pandas.DataFrame(0,index=meaning_space.schemata(), columns=signal_space.schemata())  
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta

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
            signal_generalization = list(self.signal_space.generalize(signal))
            meaning_generalization = list(self.meaning_space.generalize(meaning))

            self.matrix                                   += self.delta
            for signal_schema in signal_generalization:
                self.matrix.loc[:,signal_schema]          += (self.gamma - self.delta)

            for meaning_schema in meaning_generalization:
                self.matrix.loc[meaning_schema,:]         += (self.beta - self.delta)

            for signal_schema in signal_generalization:
                for meaning_schema in meaning_generalization:
                    self.matrix.loc[meaning_schema,signal_schema] += self.alpha - self.beta - self.gamma + self.delta
 

    def hear (self, signal):
        """
        Return the optimal meaning for a signal
        """
        meanings = self.meaning_space.meanings()
        winners = []
        maxscore = 0
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
        maxscore = 0
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
        _Learner.think(signal)
        return self.meaning_space.sample(number)
                    
    def teach(self,number):
        #        pdb.set_trace()
        _Learner.teach(number)
        thoughts = self.think(number)
        lessons = [ [self.speak(thought),thought] for thought in thoughts ]
        return lessons
            

if __name__ == "__main__":
    import doctest
    doctest.testmod()
