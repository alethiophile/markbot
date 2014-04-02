#!/usr/bin/python3

import random, re, sys
from collections import Counter
try:
    import json, lmdb
except ImportError:
    json = lmdb = None

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
    def __init__(self, clen=3, db=None):
        """Initialize the generator for a given chain length. The longer the chain
        length, the more similar generated output will be to training input.
        Reasonable values vary; for word-based text generation on relatively
        short inputs, 3-5 is good, though 5 may end up generating mostly direct
        quotes. Letter-based generation will need larger values.

        This object can use an LMDB database, if passed it. If desired, 'db'
        should be a tuple of (env, db), where these are the result of
        lmdb.open() and env.open_db() respectively. (MarkovGenerator does not
        want or need to care about filenames, named databases, &c.; it uses the
        handles passed. Actually doing LMDB handling is left to the caller.) The
        passed-in DB will be arbitrarily scribbled-on by the writing of keys; it
        really does need a dedicated one per generator.

        """
        self.clen = clen
        if db is None:
            self.lmdb = False
            self.db = {}
        else:
            self.lmdb = True
            self.env = db[0]
            self.db = db[1]

    def ntuples(self, ilist):
        if len(ilist) < self.clen:
            raise Exception("Not enough data")
        for i in range(len(ilist)-self.clen+1):
            r = tuple([dedup(n) for n in ilist[i:i+self.clen]])
            yield r

    def train_token(self, k, v, txn=None):
        if not self.lmdb:
            if k in self.db:
                self.db[k].append(v)
            else:
                self.db[k] = [v]
        else:
            ks = json.dumps(k, separators=(',',':')).encode()
            vd = txn.get(ks, default=b"{}")
            vs = json.loads(vd.decode())
            if v in vs:
                vs[v] += 1
            else:
                vs[v] = 1
            vsn = json.dumps(vs, separators=(',',':')).encode()
            txn.put(ks, vsn)

    def train(self, ilist, txn=None):
        """Feed the generator input to train on. ilist is an arbitrary list of data,
        probably strings or integers. Common use cases are lists of the words or
        letters of a text file, in order to generate text with a resemblance to
        that file.

        """
        if self.lmdb:
            made = False
            if txn is None:
                made = True
                txn = self.env.begin(write=True, db=self.db)
            for i in self.ntuples(ilist):
                k, v = i[:-1], i[-1]
                self.train_token(k, v, txn)
            if made:
                txn.commit()
        else:
            for i in self.ntuples(ilist):
                k, v = i[:-1], i[-1]
                self.train_token(k, v)

    def get_statelist(self):
        if not self.lmdb:
            return list(self.db.keys())
        else:
            with self.env.begin(db=self.db) as txn:
                cursor = txn.cursor()
                i = cursor.iternext(values=False)
                return list(i)

    def get_state_toks(self, state, txn=None):
        if not self.lmdb:
            return self.db[state]
        else:
            ks = json.dumps(state, separators=(',',':'))
            v = txn.get(ks, default=b"{}")
            vs = json.loads(v.decode())
            if not vs:
                raise KeyError("Not in database")
            rv = []
            for v in vs:
                rv.extend([v] * vs[v])
            return rv

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
            state = random.choice(self.get_statelist())
        else:
            state = istate
        for i in state:
            yield i
        n -= len(state)
        if self.lmdb:
            txn = self.env.begin(db=self.db)
        else:
            txn = None
        for i in range(n):
            try:
                w = random.choice(self.get_state_toks(state, txn))
            except KeyError:
                break
            state = state[1:] + (w,)
            yield w
