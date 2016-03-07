import warnings
import pandas
import pdb
import ilmpy
import random
from enum import Enum

class ActionMode(Enum):
    neither = 0
    hear = 1
    speak = 2
    

class _Learner ():
    """
    This is a private base class 
    """
    def __init__(self, signal_space, meaning_space):
        self.signal_space = signal_space
        self.meaning_space = meaning_space
        self.mode = ActionMode.neither

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
        self.mode = ActionMode.hear
    
    def think (self, number):
        """
        Returns a list of a specified number of random meanings
        """
        if (number < 0 or (number != floor(number))):
            raise ValueError("Parameter number must be an integer >= 0. You passed %f" % (number))

    def speak (self, meanings):
        """
        Produce a list of signals corresponding to a list of meanings
        """
        if (number < 0 or (number != floor(number))):
            raise ValueError("Parameter number must be an integer >= 0. You passed %f" % (number))
        self.mode = ActionMode.speak

    def teach (self, meanings):
        """
        Produce a list of signal-meaning pairs corresponding to a list of meanings
        """
        if (number < 0 or (number != floor(number))):
            raise ValueError("Parameter number must be an integer >= 0. You passed %f" % (number))


class AssociationMatrixLearner (_Learner):
    """
    This class implements the original Smith-Kirby ILM

    """
    def __init__(self,signal_space, meaning_space, alpha=1, beta=-1, gamma=-1, delta=0):
        _Learner.__init__(self, signal_space, meaning_space)
        self.matrix = pandas.DataFrame(index=meaning_space.meanings(), columns=signal_space.signals())  
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        
    def learn(self,data):
        #        pdb.set_trace()
        _Learner.learn(data)
        for datum in data:
            signal = datum[0]
            meaning = datum[1]
            signal_analysis = self.signal_space.analyze(signal)
            meaning_analysis = self.meaning_space.analyze(meaning)
            for signal_schema in signal_analysis:
                for meaning_schema in meaning_analysis:
                    self.matrix                                   + delta
                    self.matrix.loc[signal_schema,:]              + (beta - delta)                     
                    self.matrix.loc[:,meaning_schema]             + (gamma - delta)
                    self.matrix.loc[signal_schema,meaning_schema] + (alpha - beta - gamma - delta) 

    def score(self,mode,signal_schema,meaning_schema):
        if self.mode == ActionMode.hear:
            weight  = self.signal_space.weight(signal_schema)
        else if self.mode == ActionMode.speak:
            weight = self.meaning_space.weight(meaning_schema)
        else:
            raise ValueError("Parameter number must be an integer >= 0. You passed %f" % (number))
        strength = self.matrix.loc[signal_schema,meaning_schema]
        return weight * strength
        
    def hear (self, signal):
        """
        Returns the optimal meaning for a signal
        """
        _Learner.hear(signal)
        signal_analysis = self.signal_space.analyze(signal)
        meanings = self.meaning_space.meanings()
        winners = []
        maxscore = 0
        for m in meanings:
            meaning_analysis = self.meaning_space.analyze(meaning)
            score = 0
            for signal_schema in signal_analysis:
                for meaning_schema in meaning_analysis:
                    score += self.score("hear",signal_schema,meaning_schema)
            if (score > maxscore):
                maxscore = score
                winners = [m]
            elif (score == maxscore):
                winners.append(m)
        self.mode = ActionMode.neither
        if (len(winners) == 1):
            return winners[0]
        else:
            return random.choice(winners)        

    def think (self, number):
        """
        Returns a list of a specified number of random meanings
        """
        _Learner.hear(signal)
        
                    
    def teach(self,number):
        #        pdb.set_trace()
        _Learner.teach(number)
        

        for datum in data:
            signal = datum[0]
            meaning = datum[1]
            signal_analysis = self.signal_space.analyze(signal)
            meaning_analysis = self.meaning_space.analyze(meaning)
            for signal_schema in signal_analysis:
                for meaning_schema in meaning_analysis:
                    self.matrix                                   + delta
                    self.matrix.loc[signal_schema,:]              + (beta - delta)                     
                    self.matrix.loc[:,meaning_schema]             + (gamma - delta)
                    self.matrix.loc[signal_schema,meaning_schema] + (alpha - beta - gamma - delta) 
            

if __name__ == "__main__":
    import doctest
    doctest.testmod()
