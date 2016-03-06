import warnings
import pandas
import pdb

class _Learner ():
    """
    This is a private base class 
    """
    def __init__(self):
        pass

    def learn (self, data):
        """
        Learn associations from a list of signal-meaning pairs
        """
        pass

    def teach (self, number):
        """
        Returns a list of a specified number of signal-meaning pairs
        """
        if (number < 0 or (number != floor(number))):
            raise ValueError("Parameter number must be an integer >= 0. You passed %f" % (number))

    def speak (self, number):
        """
        Returns a list of a specified number of signals
        """
        if (number < 0 or (number != floor(number))):
            raise ValueError("Parameter number must be an integer >= 0. You passed %f" % (number))


    def think (self, number):
        """
        Returns a list of a specified number of random meanings
        """
        if (number < 0 or (number != floor(number))):
            raise ValueError("Parameter number must be an integer >= 0. You passed %f" % (number))
    
    def hear (self, signal):
        """
        Returns the meaning for a signal
        """
        if (signal not in self.signal_space.signals() ):
            raise ValueError("Signal unrecognized. You passed %s" % (signal))
    
                


class AssociationMatrixLearner (_Learner):
    """
    This class implements the original Smith-Kirby ILM

    """
    def __init__(self):
        _Learner.__init__(self, signal_space, meaning_space, alpha=1, beta=-1, gamma=-1, delta=0)
        self.matrix = pandas.DataFrame(index=meaning_space.meanings(), columns=signal_space.signals())  
        self.signal_space = signal_space
        self.meaning_space = meaning_space
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
