#!/usr/bin/python3.3

# A bot that does Markov-chain training on everything it hears, then spits out
# lines on command. Maintains separate chains for each channel it joins and each
# user it hears.

import markov
import re, sys, signal, os, argparse
import pickle as pickle
import lmdb
from irc.bot import SingleServerIRCBot
from multiprocessing import Process, Queue
import builtins

# Utility functions for chains.
def say_from(chain, l=250):
    clen = chain.clen
    t = tuple(([''] * (clen - 2)) + ['v'])
    m = ''.join(chain.generate(l, t))[1:]
    return m

def say_about(chain, about, l=250):
    clen = chain.clen
    kl = clen - 1
    s = (['v'] + list(about))[-kl:]
    s = ([''] * (kl - len(s))) + s
    m = ''.join(chain.generate(l, tuple(s)))[1:]
    if not m.startswith(about):
        m = about[:-(kl-1)] + m
    if m == about:
        if about[0].islower():
            m = say_about(chain, about[0].upper() + about[1:], l)
        else:
            m = "I don't know anything about that."
    return m.replace("\n", "")

def learn(message, chain, txn=None):
    clen = chain.clen
    l = ([''] * (clen - 2)) + list(message)
    try:
        chain.train(l, txn)
    except Exception:
        pass

def trainlines(fname, c):
    """Trains the given chain with lines from the given file."""
    l = file(fname).readlines()
    for i in l:
        learn(i, c)

def readlogs(fn):
    """Read logs, do the right thing with them. Handles either irssi or weechat
    logs.

    """
    f = open(fn)
    for n,i in enumerate(f):
        if i.startswith('---'):
            continue
        if re.match(r"^\d\d:\d\d:\d\d", i):
            o = re.match(r"\d\d:\d\d:\d\d\t([^\t]+)\t(.+)", i)
            if o is None: continue
            a = o.groups()
            if a[0] == " *":
                who, msg = a[1].split(' ', 1)
                msg = "a" + msg
            elif re.match(r"<.*>", a[0]):
                if a[0][1] in "+%@~":
                    who = a[0][2:-1]
                else:
                    who = a[0][1:-1]
                msg = "v" + a[1]
            else:
                continue
        elif re.match(r"\d\d:\d\d", i):
            o = re.match(r"\d\d:\d\d ( \*|< [^ ]+>|[^ ]+) (.+)", i)
            if o is None: continue
            a = o.groups()
            if a[0] == "-!-":
                continue
            elif a[0].startswith("<"):
                who = a[0][2:-1]
                msg = "v" + a[1]
            elif a[0] == " *":
                who, msg = a[1].split(' ', 1)
                msg = "a" + msg
        yield (who, msg)

def fold_string_indiscriminately(s, n=80):
    """Folds a string (insert line-breaks where appropriate, to format
    on a display of no more than n columns) indiscriminately, meaning
    lose all existing whitespace formatting. This is the equivalent of
    doing an Emacs fill-paragraph on the string in question, though it
    doesn't break around double linefeeds like that function does."""
    l = s.split()
    rv = []
    cl = 0
    rl = []
    for i in l:
        if cl + len(i) + 1 < n:
            rl.append(i)
            cl += len(i) + 1
        else:
            rv.append(rl)
            rl = [i]
            cl = len(i)
    rv.append(rl)
    return '\n'.join([' '.join(i) for i in rv])

class MarkovBot(SingleServerIRCBot):
    def __init__(self, channel, nick, server, port=6667, clen=9, db=None, nspw=None):
        """Arguments are straightforward. Exception: if 'db' is not None, it signifies
        desire to use an LMDB database as backing storage instead of memory; its
        value should be a string containing the name of the database directory.

        """
        SingleServerIRCBot.__init__(self, [(server, port)], nick, nick)
        if channel:
            self.clist = [channel]
        else:
            self.clist = []
        self.clen = clen
        self.nspw = nspw
        self.chains = {}
        self.mlen = 480
        if db is not None:
            self.env = lmdb.open(db, map_size=4294967296, max_dbs=4096)
            with self.env.begin() as txn:
                c = txn.cursor()
                for i in c.iternext(values=False):
                    db = self.env.open_db(name=i)
                    i = i.decode()
                    self.chains[i] = markov.MarkovGenerator(self.clen, (self.env, db))
        else:
            self.env = None

    def __enter__(self):
        return self

    def __exit__(self, etype, exval, etb):
        if self.env is not None:
            self.env.close()
        self.die('')
        return False

    def say(self, chain):
        m = say_from(chain, self.mlen)
        nstr = "{}: ".format(self.connection.get_nickname())
        if m.startswith(nstr):
            m = m[len(nstr):]
        return m.replace("\n", "")

    def on_nicknameinuse(self, c, e):
        on = c.get_nickname()
        print(("Nick {} in use".format(on)))
        c.nick(on + "_")

    def on_welcome(self, c, e):
        for i in self.clist:
            c.join(i)
        if self.nspw:
            c.privmsg("NickServ", "identify {}".format(self.nspw))

    def get_chain(self, arg):
        if not arg in self.chains:
            if self.env is None:
                self.chains[arg] = markov.MarkovGenerator(self.clen)
            else:
                argb = arg.encode() if type(arg) == str else arg
                db = self.env.open_db(name=argb)
                self.chains[arg] = markov.MarkovGenerator(self.clen, (self.env, db))
        return self.chains[arg]

    def send_to(self, to, msg):
        l = fold_string_indiscriminately(msg, 450).split('\n')
        for i in l:
            self.connection.privmsg(to, i)

    def on_privmsg(self, conn, ev):
        message = ev.arguments[0]
        rnick = ev.source.split('!')[0]
        l = message.split()
        if l[0] == 'talk':
            c = self.get_chain(l[1])
            conn.privmsg(rnick, self.say(c))
        elif l[0] == 'dump':
            self.dump_db(l[1])
        elif l[0] == 'load':
            self.load_db(l[1])
        elif l[0] == 'logs':
            self.send_to(rnick, "Training logs (file {}, channel {})".format(l[2], l[1]))
            self.train_logs(l[2], l[1])
            self.send_to(rnick, "Done")
        elif l[0] == 'join':
            self.clist.append(l[1])
            conn.join(l[1])
        elif l[0] == 'leave':
            conn.part(l[1])
        elif l[0] == 'chains':
            self.send_to(rnick, ', '.join(list(self.chains.keys())))
            if len(self.chains) == 0:
                self.send_to(rnick, "No chains")
        elif l[0] == 'version':
            self.send_to(rnick, "Markov bot v0.6")

    def on_pubmsg(self, conn, ev):
        target = ev.target
        message = ev.arguments[0]
        nick = conn.get_nickname()
        rnick = ev.source.split('!')[0]
        c = self.get_chain(target)
        learn("v" + message, c)
        uc = self.get_chain(rnick)
        learn("v" + message, uc)
        if nick.lower() in message.lower():
            m = re.search("[Tt]alk like ([#\w-]+)", message)
            if m:
                c = self.get_chain(m.group(1))
            m = re.search("[Aa]bout (.+?)[.?!]?\s*$", message)
            if m:
                conn.privmsg(target, say_about(c, m.group(1)))
                return
            s = self.say(c)
            conn.privmsg(target, s)

    def dump_db(self, outfile):
        dobj = (self.chains, self.clen, self.clist, self.mlen)
        with open(outfile, 'wb') as o:
            pickle.dump(dobj, o, -1)

    def load_db(self, infile):
        with open(infile, 'rb') as i:
            self.chains, self.clen, self.clist, self.mlen = pickle.load(i)

    def train_logs(self, infile, channel):
        c = self.get_chain(channel)
        for i in readlogs(infile):
            learn(i[1], c)
            if len(i[0]) == 0:
                continue
            uc = self.get_chain(i[0])
            learn(i[1], uc)

    def on_action(self, conn, ev):
        message = ev.arguments[0]
        target = ev.target
        nick = conn.get_nickname()
        rnick = ev.source.split('!')[0]
        c = self.get_chain(target)
        learn("a" + message, c)
        uc = self.get_chain(rnick)
        learn("a" + message, uc)
        if nick.lower() in message.lower():
            conn.privmsg(target, self.say(c))

def do_training(source, chain, db, clen=9):
    """Take a source (an iterator over lines) and a chain (name of chain to train
    into), and train the lines into the chain. If the source yields tuples, they
    will be assumed to be 2-tuples (chain, line), and each line will be trained
    into both the chain named as an argument and the chain in the tuple.

    """
    def write_items(db, q, clen):
        env = lmdb.open(db, map_size=4294967296, max_dbs=4096)
        chains = {}
        with env.begin() as txn:
            c = txn.cursor()
            for i in c.iternext(values=False):
                db = env.open_db(name=i)
                i = i.decode()
                chains[i] = markov.MarkovGenerator(clen, (env, db))

        lines = {}
        names = []

        def get_chain(arg):
            if not arg in chains:
                argb = arg.encode() if type(arg) == str else arg
                db = env.open_db(name=argb)
                chains[arg] = markov.MarkovGenerator(clen, (env, db))
            return chains[arg]

        for i in iter(q.get, None):
            if len(i[0]) == 0:
                print("skipped {}".format(i))
                continue
            if not i[0] in lines:
                lines[i[0]] = []
                if not i[0] in names:
                    names.append(i[0])
                    print(i[0])
            lines[i[0]].append(i[1])
            if len(lines[i[0]]) >= 800:
                c = get_chain(i[0])
                with env.begin(write=True, db=c.db) as txn:
                    for j in lines[i[0]]:
                        learn(j, c, txn)
                del lines[i[0]]

        for i in lines.keys():
            print("Finishing {}".format(i))
            c = get_chain(i)
            with env.begin(write=True, db=c.db) as txn:
                for j in lines[i]:
                    learn(j, c, txn)

    q = Queue(10000)
    p = Process(target=write_items, args=(db,q,clen))
    p.start()

    for n,i in enumerate(source):
        if type(i) == builtins.tuple:
            q.put(i)
            q.put((chain, i[1]))
        else:
            q.put((chain, i))
        print(n, end='\r')
    print('', end='\n')

    q.put(None)
    q.close()
    p.join()

def main():
    parser = argparse.ArgumentParser(description="IRC bot that trains Markov chains and runs their output on demand")
    parser.add_argument("server", help="IRC server to connect")
    parser.add_argument("nick", help="Nickname to use")
    parser.add_argument("-p", "--port", type=int, help="Port to connect to", default=6667)
    parser.add_argument("-d", "--database-file", help="Database filename to use; if unspecified, no database", default=None)
    parser.add_argument("-c", "--channel-join", help="Channel to join on startup", default=None)
    parser.add_argument("-t", "--train-file", help="Train database (faster than online)", default=None)
    parser.add_argument("-l", "--train-logs", action="store_true", help="Train a logs file", default=False)
    parser.add_argument("-r", "--train-chain", help="Chain to train into", default=None)
    parser.add_argument("-w", "--password", help="Nickserv password", default=None)
    a = parser.parse_args()

    if a.train_file is not None:
        if not a.train_chain:
            print("Error: need chain", file=sys.stderr)
            sys.exit(1)
        if not a.database_file:
            print("Error: need DB", file=sys.stderr)
            sys.exit(1)
        if a.train_logs:
            it = readlogs(a.train_file)
        else:
            it = open(a.train_file, 'r')
        do_training(it, a.train_chain, a.database_file)
    else:
        with MarkovBot(a.channel_join, a.nick, a.server, a.port, db=a.database_file, nspw=a.password) as bot:
            bot.start()

if __name__=="__main__":
    main()
