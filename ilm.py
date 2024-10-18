#! /usr/bin/python
from __future__ import division 
from __future__ import print_function
from optparse import OptionParser, OptionValueError
#from types import FloatType
import ilmpy
from ilmpy.argument_parser import ILM_Parser
import time
import sys
import numpy
import random
import pdb

starttime = time.time()
if __name__ == "__main__":
    version = 0.2
    prog = 'ilm'
    usage = '''usage: %prog [options] <SIGNAL-SPACE-PATTERN> <MEANING-SPACE-PATTERN> 

Iterated Learning Models in Python (ILMpy) version 0.2
Copyright (2018) David H. Ardell
All Wrongs Reversed.
Please cite Ardell and Winter (2016) in published works using this software.

Usage:
The meaning space size must be larger than the bottleneck size set by (-I INTERACTIONS)

Examples:
ilm <SIGNAL-SPACE-PATTERN> <MEANING-SPACE-PATTERN>

ilm "[bp].[ao]"                   "(4).(3)"  # classic Smith-Kirby lattice spaces; words are e.g. "ba" and "po"
ilm "[a-z].a.[dt]"               "(16).(2)"  # compositionality
ilm "[a-c]^2"                       "(3)^3"  # "^" powers up components. Signal/meaning space sizes are 9/27
ilm "[a-z].a.[dt]"               "(16).{2}"  # unordered (set-like) meaning-space-components do not generalize
ilm "([b-d]:0.01).[aeiou]"        "(3).(4)"  # noise rate of 1% in first signal dimension
ilm "(([a-z]\[aeiou]):0.05).[ae]"   "(4)^2"  # set-complement sound-space in first dimension is noisy at 5%

THE BELOW ARE FOR FUTURE REFERENCE: generalizable sound transformations ARE NOT YET IMPLEMENTED!
ilm "(a|A).[bc]"                    "(2)^2"  # generalizable sound transformation in first signal dimension
ilm "((aeiou|AEIOU):0.01)^2"        "{2}^2"  # any sound space can be noisy
ilm "(([a-g]\[aeiou]):0.1)^2"   "{256}.(2)"  # any sound space can be powered 
'''
    parser = OptionParser(usage=usage,version='{:<3s} version {:3.1f}'.format(prog,version))
    parser.disable_interspersed_args()

    ## parser.add_option("--method", dest="method", type="choice",
    ##     	      choices=method_choices, default="association",
    ##     	      help="learning method. Choose from %s" % method_choices)

    parser.add_option("-T","--trials",
		      dest="num_trials", type="int", default=1,
		      help="set number of trials with ILM chains to simulate\n Default: %default")

    parser.add_option("-G","--generations",
		      dest="num_generations", type="int", default=10,
		      help="set number of generations (chain length)\n Default: %default")

    parser.add_option("-I","--interactions",
		      dest="num_interactions", type="int", default=10,
		      help="set number of teaching interactions (signal-meaning pairs) communicated from parent to child\n Default: %default")

    parser.add_option("-a","--alpha",
		      dest="alpha", type="float", default=1.0,
		      help="set alpha \n Default: %default")

    parser.add_option("-b","--beta",
		      dest="beta", type="float", default=0.0,
		      help="set beta\n Default: %default")

    parser.add_option("-g","--gamma",
		      dest="gamma", type="float", default=-1.0,
		      help="set gamma\n Default: %default")

    parser.add_option("-d","--delta",
		      dest="delta", type="float", default=0.0,
		      help="set delta\n Default: %default")

    parser.add_option("-e","--noise",
		      dest="noise", type="float", default=0.0,
		      help="set base signal-noise rate. Not yet implemented, specify noise through arguments instead. Default: %default")

    parser.add_option("-c","--cost",
		      dest="cost", type="float", default=0.0,
		      help="set base misunderstanding cost function. Not yet implemented, now all meanings have equal cost. Default: %default")

    parser.add_option("-s","--seed", 
		      dest="seed", type="int",  default=None,
		      help="seed random number generator. Default: %default")            

    parser.add_option("-A","--amplitude", 
		      dest="amplitude", type="float",  default=None,
		      help="Initialize agents with uniformly distributed association strengths. Range of values is 2x amplitude, centered on zero. Default: %default")     

    parser.add_option("--precision",
		      dest="precision", type="int",  default=4,
		      help="set print precision for parameter printing. Default: %default")

    parser.set_defaults(show_matrices=False, show_lessons=True, show_compositionality=False, show_accuracy=False, show_load=False, show_stats=False, show_vocabulary=False, show_final_vocabulary = False)
    parser.add_option("--show-matrices", action="store_true", dest="show_matrices", help="print internal message-signal matrices at each iteration")
    parser.add_option("--no-show-lessons", action="store_false", dest="show_lessons", help="do not print the lessons passed to new agents at each iteration")
    parser.add_option("--show-compositionality", action="store_true", dest="show_compositionality", help="print compositionality at each iteration")
    parser.add_option("--show-accuracy", action="store_true", dest="show_accuracy", help="print communicative accuracy at each iteration")
    parser.add_option("--show-load", action="store_true", dest="show_load", help="print functional load by signal position at each iteration")
    parser.add_option("--show-stats", action="store_true", dest="show_stats", help="print all statistics at each iteration")
    parser.add_option("--show-vocabulary", action="store_true", dest="show_vocabulary", help="print the signal for each meaning at each iteration")
    parser.add_option("--show-final-vocabulary", action="store_true", dest="show_final_vocabulary", help="print the signal for each meaning at the end of each chain")

    myargv = sys.argv
    (options, args) = parser.parse_args()
    if len(args) != 2:
    	parser.error("expects two arguments")

    arg_string = '{} {}'.format(*args)
    ilm_parser = ILM_Parser()
    try:
        (signal_space,meaning_space) = ilm_parser.parse(arg_string)
    except ValueError:
        print('\n')
        print(usage)
        print('\n{}: syntax error invalid arguments to ilm: {}\n'.format(prog,arg_string))
        sys.exit(0)


    program_args = [meaning_space, signal_space, options.alpha, options.beta, options.gamma, options.delta]
    program_kwargs = {}

    if options.seed is not None:
        numpy.random.seed(options.seed)
        random.seed(options.seed)

    if options.amplitude is not None:
        program_kwargs['amplitude'] = options.amplitude

    observables = ilmpy.observables.Observables(show_matrices                  = options.show_matrices,
                                                show_lessons                   = options.show_lessons,
                                                show_vocabulary                = options.show_vocabulary,
                                                show_final_vocabulary          = options.show_final_vocabulary,
                                                show_compositionality          = options.show_compositionality,
                                                show_accuracy                  = options.show_accuracy,
                                                show_load                      = options.show_load,
                                                show_stats                     = options.show_stats,
                                                print_precision                = options.precision)

    program_kwargs['observables'] = observables
                                                
    print('# {:<3s} version {:3.1f}'.format(prog,version))
    print('# Copyright (2016) David H. Ardell.')
    print('# All Wrongs Reversed.')
    print('#')
    print('# Please cite Ardell and Winter (2016) in published works using this software.')
    print('#')
    print('# execution command:')
    print('# '+' '.join(myargv))
    print('#')

    for trial in range(options.num_trials):
        parent = ilmpy.learners.AssociationMatrixLearner(*program_args,**program_kwargs)
        if trial == 0:
            parent.print_parameters()
            if options.seed is not None:
                print('# seed: {}'.format(options.seed))                
            if options.amplitude is not None:
                print('# amplitude: {}'.format(options.amplitude))
            print('# bottleneck: {}\n# iterations: {}\n# trials: {}'.format(options.num_interactions,options.num_generations,options.num_trials))
            print('# ')
            parent.print_observables_header()
        for generation in range(options.num_generations):
            print('# Trial {} Iteration {}'.format(trial,generation))
            child = parent.spawn()
            #print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx")
            lessons = parent.teach(options.num_interactions)
            child.learn(lessons)
            child.print_observables()
            parent = child
        
print("# Run time (minutes): ",round((time.time()-starttime)/60,3))
                    
