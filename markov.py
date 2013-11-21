#!/usr/bin/python
# coding=UTF-8

import random, re, sys
from collections import Counter

toks = [] # global token deduplicator

def dedup(tok):
    """Takes a token and deduplicates it, either by returning the previous version or by adding it to the list."""
    try:
        n = toks.index(tok)
    except ValueError:
        toks.append(tok)
        return tok
    else:
        return toks[n]

class MarkovGenerator:
    def __init__(self, clen=3):
        """Initialize the generator for a given chain length. The longer the chain
        length, the more similar generated output will be to training
        input. Reasonable values vary; for word-based text generation on
        relatively short inputs, 3-5 is good, though 5 may end up generating
        mostly direct quotes. Letter-based generation will need larger values.

        """
        self.clen = clen
        self.db = {}

    def ntuples(self, ilist):
        if len(ilist) < self.clen:
            raise Exception("Not enough data")
        for i in xrange(len(ilist)-self.clen+1):
            r = tuple([dedup(n) for n in ilist[i:i+self.clen]])
            yield r

    def train(self, ilist):
        """Feed the generator input to train on. ilist is an arbitrary list of data,
        probably strings or integers. Common use cases are lists of the words or
        letters of a text file, in order to generate text with a resemblance to
        that file.

        """
        for i in self.ntuples(ilist):
            k, v = i[:-1], i[-1]
            if k in self.db:
                self.db[k].append(v)
            else:
                self.db[k] = [v]

    def generate(self, n, istate=None):
        """Generate output. The result is a generator that will yield at most n
        tokens. (Fewer may be generated, if the generator achieves a terminal
        state. Each individual training creates a new terminal state, making
        this more likely.) If istate is provided, then that will be the initial
        state; otherwise, it will be chosen randomly from the database keys.

        """
        if n < self.clen:
            raise Exception("Not long enough")
        if istate == None:
            state = random.choice(self.db.keys())
        else:
            state = istate
        for i in state:
            yield i
        n -= len(state)
        for i in range(n):
            try:
                w = random.choice(self.db[state])
            except KeyError:
                break
            state = state[1:] + (w,)
            yield w
