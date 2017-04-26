from __future__ import print_function
from __future__ import division 
import ply.lex as lex
import ply.yacc as yacc
import os
#import ilmpy
import ilmpy.signal_spaces as signal_spaces
import ilmpy.meaning_spaces as meaning_spaces



#%prog <SIGNAL-SPACE-PATTERN> <MEANING-SPACE-PATTERN> 
    # signals are strings, meanings are vectors of numbers or tuples of numbers and grah 

    
    # eventually: {1024}^3.((singular:0.1,plural:0.2)noun:0.3,(past:0.2,present:0.1)verb:0.4)


class ILM_Parser:
    """
    Base class for a lexer/parser that has the rules defined as methods

    >>> p = ILM_Parser(debug=1)

    >>> args = '[a-z]^2 (4)^2'                              # small lattices
    >>> (signal_space,meaning_space) = p.parse(args)

    >>> args = '[a-g]^3 {3}.(4).(2)'                        # unordered (set-like) meaning-spaces 
    >>> (signal_space,meaning_space) = p.parse(args)

    >>> args = '([b-d]:0.01).[aeiou] (3).(4)'               # noiserates
    >>> (signal_space,meaning_space) = p.parse(args)

    >>> args = '(([a-z]\[aeiou]):0.05).[aeiou] (4).(2)^2'   # noiserates can go any sound-space
    >>> (signal_space,meaning_space) = p.parse(args)

    >>> args = '(a|A).[bc] (2)^2'                           # generalizable transformation sound-space
    >>> (signal_space,meaning_space) = p.parse(args)

    >>> args = '((aeiou|AEIOU):0.01)^2 {2}^2'               # transformation sound-space with noise
    >>> (signal_space,meaning_space) = p.parse(args)
       
    >>> args = '([a-g]\[aeiou])^2.(aeiou|AEIOU).(bd|pt) (8).(5)' # set-complements
    >>> (signal_space,meaning_space) = p.parse(args)

    >>> args = '(([a-g]\[aeiou]):0.1)^2 {256}.(2)'            # with noise and powered
    >>> (signal_space,meaning_space) = p.parse(args)

    """

    def __init__(self, **kw):
        self.debug = False
        self.names = { }
        try:
            modname = os.path.split(os.path.splitext(__file__)[0])[1] + "_" + self.__class__.__name__
        except:
            modname = "parser"+"_"+self.__class__.__name__
        self.debugfile = modname + ".dbg"
        self.tabmodule = modname + "_" + "parsetab"
        #print self.debugfile, self.tabmodule

        # Build the lexer and parser
        lex.lex(module=self)#, debug=self.debug)
        self.yacc = yacc.yacc(module=self,
                              debug=self.debug,
                              debugfile=self.debugfile,
                              tabmodule=self.tabmodule)
        
    def parse(self, args):
        return self.yacc.parse(args)#, debug=True)
        
    tokens = (
        'LPAREN',
        'LSQUARE',
        'LETTER',
        'ALPHASTRING',
        'DASH',
        'RSQUARE',
        'BACKSLASH',
        'LBRACE',
        'INTEGER',
        'RBRACE',
        'DOT',
        'RPAREN',
        'COLON',
        'FLOAT',
        'PIPE',
        'SPACE',
        'HAT',
        )
    #    'COMMA'


    # Regular expression rules for simple tokens
    t_LPAREN    = r'\('
    t_LSQUARE   = r'\['
    t_DASH      = r'\-'
    t_RSQUARE   = r'\]'
    t_BACKSLASH = r'\\'
    t_LBRACE    = r'\{'
    t_RBRACE    = r'\}'
    t_DOT       = r'\.'
    t_RPAREN    = r'\)'
    t_COLON     = r':'
    t_PIPE      = r'\|'
    t_HAT       = r'\^'
    #t_COMMA     = r','

    def t_FLOAT(self,t):
        r'[0-9]+\.[0-9]+'
        t.value = float(t.value)    
        return t

    def t_INTEGER(self,t):
        r'\d+'
        t.value = int(t.value)    
        return t

    def t_ALPHASTRING(self,t):
        r'[a-zA-Z][a-zA-Z]+'
        return t

    def t_SPACE(self,t):
        r'\s+'
        return t

    def t_LETTER(self,t):
        r'[a-zA-Z]'
        return t

    # Error handling rule
    def t_error(self,t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    # arguments        : signal-space meaning-space

    # signal-space     : signal-component DOT signal-space
    # signal-space     : signal-component HAT INTEGER DOT signal-space
    # signal-space     : signal-component HAT INTEGER 
    # signal-space     : signal-component 

    # signal-component : LPAREN sound-space COLON noise-rate RPAREN
    #                  | sound-space

    # sound-space      : LPAREN ALPHASTRING PIPE ALPHASTRING RPAREN # transform
    # sound-space      : LPAREN LETTER PIPE LETTER RPAREN # transform
    # sound-space      | LPAREN char-set BACKSLASH char-set RPAREN  # set-difference
    # sound-space      | char-set

    # char-set         : LSQUARE ALPHASTRING RSQUARE 
    #                  | LSQUARE range RSQUARE
    #                  | LETTER

    # range            : LETTER DASH LETTER

    # noise-rate       : FLOAT

    # meaning-space     : meaning-component DOT meaning-space
    # meaning-space     : meaning-component HAT INTEGER  DOT meaning-space
    # meaning-space     : meaning-component HAT INTEGER 
    # meaning-space     : meaning-component 
    # meaning-component : LPAREN INTEGER RPAREN
    # meaning-component : LBRACE INTEGER RBRACE 

    ## precedence = (
    ##     ('right', 'SPACE'),
    ## )

    def p_arguments(self,p):
        'arguments : signal-space SPACE meaning-space'
        p[0] = [p[1],p[3]]

    def p_signal_space_power_dot(self,p):
        'signal-space : signal-space DOT signal-component HAT INTEGER'
        for i in range(p[5]):
            p[1].add_component(p[3])
        p[0] = p[1]

    def p_signal_space_dot(self,p):
        'signal-space : signal-space DOT signal-component'
        p[1].add_component(p[3])
        p[0] = p[1]

    def p_signal_space_power(self,p):
        'signal-space : signal-component HAT INTEGER'
        p[0] = signal_spaces.WordSignalSpace()
        for i in range(p[3]):
            p[0].add_component(p[1])

    def p_signal_space(self,p):
        'signal-space : signal-component'
        p[0] = signal_spaces.WordSignalSpace()
        p[0].add_component(p[1])

    def p_signal_component_noise(self,p):
        'signal-component : LPAREN sound-space COLON noise-rate RPAREN'
        p[2].set_noiserate(p[4])
        p[0] = p[2]

    def p_signal_component(self,p):
        'signal-component : sound-space'
        p[0] = p[1]

    def p_sound_space_transform(self,p):
        'sound-space :  LPAREN ALPHASTRING PIPE ALPHASTRING RPAREN'
        p[0] = signal_spaces.TransformSignalComponent( p[2], p[4])

    def p_sound_space_transform_letter(self,p):
        'sound-space :  LPAREN LETTER PIPE LETTER RPAREN'
        p[0] = signal_spaces.TransformSignalComponent( p[2], p[4])

    def p_sound_space_difference(self,p):
        'sound-space : LPAREN char-set BACKSLASH char-set RPAREN'
        p[0] = signal_spaces.SignalComponent( p[2] - p[4] )

    def p_sound_space_char_set(self,p):
        'sound-space : char-set'
        p[0] = signal_spaces.SignalComponent( p[1] )

    def p_char_set_string(self,p):
        'char-set : LSQUARE ALPHASTRING RSQUARE'
        p[0] = set(p[2])

    def p_char_set_range(self,p):
        'char-set : LSQUARE range RSQUARE'
        p[0] = set(p[2])

    def p_char_set_letter(self,p):
        'char-set : LETTER'
        p[0] = set(p[1])

    def p_range(self,p):
        'range : LETTER DASH LETTER'
        p[0] = ''.join([chr(c) for c in range(ord(p[1]), ord(p[3])+1)])

    def p_noise_rate(self,p):
        'noise-rate : FLOAT'
        p[0] = p[1]

    def p_meaning_space_power_dot(self,p):
        'meaning-space : meaning-space DOT meaning-component HAT INTEGER'
        for i in range(p[5]):
            p[1].add_component(p[3])
        p[0] = p[1]

    def p_meaning_space_dot(self,p):
        'meaning-space : meaning-space DOT meaning-component'
        p[1].add_component(p[3])
        p[0] = p[1]

    def p_meaning_space_power(self,p):
        'meaning-space : meaning-component HAT INTEGER'
        p[0] = meaning_spaces.CombinatorialMeaningSpace()
        for i in range(p[3]):
            p[0].add_component(p[1])

    def p_meaning_space(self,p):
        'meaning-space : meaning-component'
        p[0] = meaning_spaces.CombinatorialMeaningSpace()
        p[0].add_component(p[1])


    def p_meaning_component_range(self,p):
        'meaning-component : LPAREN INTEGER RPAREN'
        p[0] = meaning_spaces.OrderedMeaningComponent(p[2])

    def p_meaning_component_set(self,p):
        'meaning-component : LBRACE INTEGER RBRACE'
        p[0] = meaning_spaces.UnorderedMeaningComponent(p[2])

    # Error rule for syntax errors
    def p_error(self,p):
        raise ValueError

if __name__ == "__main__":
    import doctest
    doctest.testmod()
