"""
Control and select output from CMCpy simulations
"""

import numpy

class Observables():
    """

    """
    def __init__(self, show_codes = True, show_messages = False, show_initial_parameters = True, show_matrix_parameters = False, show_fitness_statistics = False, show_code_evolution_statistics = False, show_frozen_results_only = False, print_precision = 6, show_all = False):
        self.show_initial_parameters = show_initial_parameters
        self.show_matrix_parameters = show_matrix_parameters
        self.show_codes = show_codes
        self.show_messages = show_messages
        self.show_fitness_statistics = show_fitness_statistics
        self.show_code_evolution_statistics = show_code_evolution_statistics
        self.show_frozen_results_only = show_frozen_results_only
        self.print_precision = print_precision
        self.show_all = show_all

if __name__ == "__main__":
    import doctest
    doctest.testmod()
