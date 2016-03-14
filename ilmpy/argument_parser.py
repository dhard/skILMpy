import ply.lex as lex
import ply.yacc as yacc
import os
import ilmpy
import signal_spaces
import meaning_spaces

class ILMParser:
    """
    Base class for a lexer/parser that has the rules defined as methods

    >>> p = ILMParser()
    >>> args = '([a-z]\[aeiou]).[aeiou] 8.2^2'
    >>> (signal_space,meaning_space) = p.yacc.parse(args)  
    """

    def __init__(self, **kw):
        self.debug = kw.get('debug', 0)
        self.names = { }
        try:
            modname = os.path.split(os.path.splitext(__file__)[0])[1] + "_" + self.__class__.__name__
        except:
            modname = "parser"+"_"+self.__class__.__name__
        self.debugfile = modname + ".dbg"
        self.tabmodule = modname + "_" + "parsetab"
        #print self.debugfile, self.tabmodule

        # Build the lexer and parser
        lex.lex(module=self, debug=self.debug)
        self.yacc = yacc.yacc(module=self,
                              debug=self.debug,
                              debugfile=self.debugfile,
                              tabmodule=self.tabmodule)
        
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
        r'[0-9]+\.[0.9]+'
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


    #%prog <SIGNAL-SPACE-PATTERN> <MEANING-SPACE-PATTERN> 
    # signals are strings, meanings are vectors of numbers or tuples of numbers and grah 

    #%prog [a-z]^8 4^2 # small lattices
    #%prog [a-z]^8 3^8
    #%prog [a-g]^3 3.4.2
    #%prog (a|A)[bc] 2^2 # generalizable transformations
    #%prog ([a-z]/[aeiou])^4.(aeiou|AEIOU).(bd|pt) 8.5.8 # set-complements
    #%prog ([a-z]/[aeiou])^4.(aeiou|AEIOU).(bd|pt) {1024}.2
    #%prog ([a-z]/[aeiou])^4.((aeiou|AEIOU):0.01).(bd|pt) {1024}^3.((singular:0.1,plural:0.2)noun:0.3,(past:0.2,present:0.1)verb:0.4)

    # arguments        : signal-space meaning-space

    # signal-space     : signal-component DOT signal-space
    # signal-space     : signal-component LBRACE INTEGER RBRACE DOT signal-space
    # signal-space     : signal-component LBRACE INTEGER RBRACE
    # signal-space     : signal-component 

    # signal-component : LPAREN sound-space COLON noise-rate RPAREN
    #                  | sound-space

    # sound-space      : LPAREN ALPHASTRING PIPE ALPHASTRING RPAREN # transform
    # sound-space      | LPAREN char-set BACKSLASH char-set RPAREN  # set-difference
    # sound-space      | char-set

    # char-set         : LSQUARE ALPHASTRING RSQUARE 
    #                  | LSQUARE range RSQUARE
    #                  | LETTER

    # range            : LETTER DASH LETTER

    # noise-rate       : FLOAT

    # meaning-space     : meaning-component DOT meaning-space
    # meaning-space     : meaning-component 
    # meaning-component : INTEGER
    # meaning-component : LBRACE INTEGER RBRACE 

    precedence = (
        ('right', 'DOT'),
    )

    def p_arguments(self,p):
        'arguments : signal-space SPACE meaning-space'
        p[0] = [p[1],p[2]]

    def p_meaning_space_dot(self,p):
        'meaning-space : meaning-component DOT meaning-space'
        p[3].add_component(p[1])
        p[0] = p[3]

    def p_meaning_space_power_dot(self,p):
        'meaning-space : meaning-component HAT INTEGER DOT meaning-space'
        for i in range(p[3]):
            p[5].add_component(p[1])
        p[0] = p[5]

    def p_meaning_space_power(self,p):
        'meaning-space : meaning-component HAT INTEGER'
        p[0] = meaning_spaces.CombinatorialMeaningSpace()
        for i in range(p[3]):
            p[0].add_component(p[1])

    def p_meaning_space(self,p):
        'meaning-space : meaning-component'
        p[0] = meaning_spaces.CombinatorialMeaningSpace()
        p[0].add_component(p[1])

    def p_meaning_component_integer(self,p):
        'meaning-component : INTEGER'
        p[0] = meaning_spaces.OrderedMeaningComponent(p[1])

    def p_meaning_component_set(self,p):
        'meaning-component : LBRACE INTEGER RBRACE'
        p[0] = meaning_spaces.UnorderedMeaningComponent(p[2])

    def p_signal_space_dot(self,p):
        'signal-space : signal-component DOT signal-space'
        p[3].add_component(p[1])
        p[0] = p[3]

    def p_signal_space(self,p):
        'signal-space : signal-component'
        p[0] = signal_spaces.WordSignalSpace()
        p[0].add_component(p[1])

    def p_signal_space_power_dot(self,p):
        'signal-space : signal-component HAT INTEGER DOT signal-space'
        for i in range(p[3]):
            p[5].add_component(p[1])
        p[0] = p[5]

    def p_signal_space_power(self,p):
        'signal-space : signal-component HAT INTEGER'
        p[0] = signal_spaces.WordSignalSpace()
        for i in range(p[3]):
            p[0].add_component(p[1])

    def p_signal_component_noise(self,p):
        'signal-component : LPAREN sound-space COLON noise-rate RPAREN'
        p[0] = p[2]
        p[0].noise_rate = p[4]

    def p_signal_component(self,p):
        'signal-component : sound-space'
        p[0] = p[1]

    def p_sound_space_transform(self,p):
        'sound-space :  LPAREN ALPHASTRING PIPE ALPHASTRING RPAREN'
        p[0] = signal_spaces.TransformSoundSpace( p[2], p[4])

    def p_sound_space_difference(self,p):
        'sound-space : LPAREN char-set BACKSLASH char-set RPAREN'
        p[0] = signal_spaces.SoundSpace( p[2] - p[4] )

    def p_sound_space_char_set(self,p):
        'sound-space : char-set'
        p[0] = signal_spaces.SoundSpace( p[1] )

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
        p[0] = ''.join([chr(c) for c in xrange(ord(p[1]), ord(p[3])+1)])

    def p_noise_rate(self,p):
        'noise-rate : FLOAT'
        p[0] = p[1]

    # Error rule for syntax errors
    def p_error(self,p):
        print("Syntax error in input!")

if __name__ == "__main__":
    import doctest
    doctest.testmod()
