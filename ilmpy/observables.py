import ilmpy

class Observables():
    """
    
    """
    def __init__(self, show_matrices=False, show_lessons=True, show_compositionality=False, show_accuracy=False, show_load=False, show_stats=False, print_precision = 6):
        self.show_matrices = show_matrices
        self.show_lessons = show_lessons
        self.show_compositionality = show_compositionality
        self.show_accuracy = show_accuracy
        self.show_load = show_load
        self.show_stats = show_stats
        self.print_precision = print_precision

if __name__ == "__main__":
    import doctest
    doctest.testmod()
