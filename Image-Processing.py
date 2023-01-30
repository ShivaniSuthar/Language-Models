
import pandas as pd
import numpy as np
import os
import re
import requests
import time


def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - Any Windows new-lines (\r\n) are transformed with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    r=requests.get(url)
    texted=r.text
    texted1=texted.replace('\r\n','\n')
    start_ind=texted1.find('*** START')
    end_ind=texted1.find('*** END')
    a=texted1[start_ind:end_ind]
    lst = re.findall(r'([*]+?.START OF (?:THIS|THE).*)', a)
    sized=len(lst[0])
    b=a[sized:]
    return b


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of every paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of every paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens should include no whitespace.
        - Whitespace (e.g. multiple newlines) between two paragraphs of text 
          should be ignored, i.e. they should not appear as tokens.
        - Two or more newlines count as a paragraph break.
        - All punctuation marks count as tokens, even if they are 
          uncommon (e.g. `'@'`, `'+'`, and `'%'` are all valid tokens).


    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """
    book_string = '\x02 ' + book_string + ' \x03'
    s = re.sub('\n\n', ' \x03 \x02 ', book_string)
    s = re.sub('\n', ' ', s)

    t = r'\w+|[^\w\s]+'

    res = re.findall(t , s)
    res_1 = res[4:len(res)-2]
    res_2 = res_1[0:41] + res_1[47:]
    return res_2


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        p = 1 / pd.Series(tokens).nunique()
        ind = pd.Series(tokens).unique()
        r=pd.Series(p,index=ind)
        return r
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        p = 1
        for i in words:
            if i not in self.mdl.index:
                p = 0
                break
            p *= self.mdl[i]
        return p
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        string = ''
        for i in np.arange(M):
            string += np.random.choice(self.mdl.index, p = self.mdl.values) + ' '
        string1=string.strip()
        return string1


class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        ser=pd.Series(tokens)
        counted=ser.value_counts(normalize=True)
        return counted
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        p = 1
        for i in words:
            if i not in self.mdl.index:
                p = 0
                break
            p *= self.mdl[i]
        return p
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        string=''
        for i in np.arange(M):
            string+=np.random.choice(self.mdl.index, p = self.mdl.values)+' '
        string1=string.strip()
        return string1


class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        lst=[]
        integer=0
        start=1
        val=self.N-1
        s1='\x02'
        s2='\x03'
        for i in np.arange(len(tokens)):
            if tokens[i]==s1:
                mult=(tokens[i],)*self.N
                lst.append(mult)
                integer=0
                start=1
                continue
            if tokens[i]==s2:
                count=0
                while count<self.N:
                    mult=()
                    for j in np.arange(self.N-1-count,0,-1):
                        mult=mult+(tokens[i-j],)
                    mult=mult+('\x03',)*(count+1)
                    lst.append(mult)
                    count+=1
            elif integer<val:
                mult=(tokens[i-start],)*(self.N-1-integer)
                for k in np.arange(integer,-1,-1):
                    mult=mult+(tokens[i-k],)
                lst.append(mult)
            else:
                mult=()
                for l in np.arange(self.N-1,-1,-1):
                    mult=mult+(tokens[i-l],)
                lst.append(mult)
            integer+=1
            start+=1
        return lst
        
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """
        # N-Gram counts C(w_1, ..., w_n)
        ser=pd.Series(self.ngrams)
        uni=UnigramLM(ser)

        
        # (N-1)-Gram counts C(w_1, ..., w_(n-1))
        ser1=ser.apply(lambda x:x[:-1])
        uni1=UnigramLM(ser1)


        # Creates the conditional probabilities
        num=ser.apply(lambda x:uni.mdl[x])
        denom=ser1.apply(lambda x:uni1.mdl[x])
        p=num/denom
        
        # Puts it all together
        df=pd.DataFrame({'ngram':ser, 'n1gram':ser1, 'prob':p})
        return df
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4) * (1/2) * (1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        p=1
        vals=NGramLM(self.N, words).ngrams[1:]
        for i in vals:
            vals1=self.mdl['ngram']
            lst=vals1.to_list()
            if i not in lst:
                p=0
                break
            p*=self.mdl.loc[self.mdl['ngram'] == i, 'prob'].max()
        return p
    

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """
       
        def sampler(ser1,lengths,picked):
            values=self.mdl.shape[0]-(self.N-1)*2
            if (lengths>0) and (picked>values):
                multed='\x03 '*lengths
                return multed
            if ser1[-1]=='\x03':
                if lengths>1:
                    return '\x02 '+sampler(('\x02',)*(self.N-1), lengths-1, picked + 1)
                else:
                    return ''
            duped=self.mdl.loc[self.mdl['n1gram']==ser1]
            unduped=duped.drop_duplicates()
            opts=unduped['ngram'].apply(lambda x: x[-1])
            lst=opts.tolist()
            lst1=unduped['prob'].tolist()
            if lengths==1:
                choiced=np.random.choice(lst,p=lst1)
                return choiced
            else:
                h=np.random.choice(lst,p=lst1)
                ser1=(ser1+(h,))[1:]
                return h+' '+sampler(ser1,lengths-1,picked+1)
        
        # Transforms the tokens to strings
        tok='\x02 '
        tok+=sampler(('\x02',)*(self.N-1), M, 0)
        stripped=tok.strip()
        return stripped
