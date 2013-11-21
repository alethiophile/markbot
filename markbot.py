#!/usr/bin/python

# A bot that does Markov-chain training on everything it hears, then spits out
# lines on command. Maintains separate chains for each channel it joins and each
# user it hears. 

import markov
import re, sys, signal, os
import cPickle as pickle
from irc.bot import SingleServerIRCBot

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
        m = "I don't know anything about that."
    return m.replace("\n", "")

def learn(message, chain):
    clen = chain.clen
    l = ([''] * (clen - 2)) + list(message)
    try:
        chain.train(l)
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
    for i in f:
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
            o = re.match(r"\d\d:\d\d ( \*|< \w+>|[^ ]+) (.+)", i)
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
    def __init__(self, channel, nick, server, port=6667, clen=9):
        SingleServerIRCBot.__init__(self, [(server, port)], nick, nick)
        if channel:
            self.clist = [channel]
        else:
            self.clist = []
        self.clen = clen
        self.chains = {}
        self.mlen = 480        

    def say(self, chain):
        m = say_from(chain, self.mlen)
        nstr = "{}: ".format(self.connection.get_nickname())
        if m.startswith(nstr):
            m = m[len(nstr):]
        return m.replace("\n", "")

    def on_nicknameinuse(self, c, e):
        on = c.get_nickname()
        print("Nick {} in use".format(on))
        c.nick(on + "_")
        
    def on_welcome(self, c, e):
        for i in self.clist:
            c.join(i)

    def get_chain(self, arg):
        if not arg in self.chains:
            self.chains[arg] = markov.MarkovGenerator(self.clen)
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
            self.send_to(rnick, ', '.join(self.chains.keys()))
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
            m = re.search("[Aa]bout (\w+)[.?!]?\s*$", message)
            if m:
                conn.privmsg(target, say_about(c, m.group(1)))
                return
            s = self.say(c).decode('utf-8')
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

def main():
    bot, server, port, nick = None, None, None, None
    def sigterm(sig, thing):
        p = os.getpid()
        fn = "mbdump-{}".format(p)
        bot.dump_db(fn)
        d = (server, port, nick, fn)
        with open('markbot-{}'.format(p), 'wb') as o:
            pickle.dump(d, o, -1)
        if sig != None:
            os._exit(1)
    if len(sys.argv) < 2:
        print "Usage: {} <server[:port]> <nick> [<channel>]".format(sys.argv[0])
        sys.exit(1)
    if len(sys.argv) < 3:
        if os.path.exists(sys.argv[1]):
            with open(sys.argv[1], 'rb') as i:
                server, port, nick, fn = pickle.load(i)
            bot = MarkovBot(None, nick, server, port)
            bot.load_db(fn)
            try:
                bot.start()
            except Exception as e:
                sigterm(None, None)
                raise e
        else:
            print "Usage: {} <server[:port]> <nick> [<channel>]".format(sys.argv[0])
            sys.exit(1)
    s = sys.argv[1].split(':', 1)
    server = s[0]
    try:
        port = int(s[1])
    except IndexError:
        port = 6667
    try:
        channel = sys.argv[3]
    except IndexError:
        channel = None
    nick = sys.argv[2]
    bot = MarkovBot(channel, nick, server, port)
    signal.signal(signal.SIGTERM, sigterm)
    try:
        bot.start()
    except Exception as e:
        sigterm(None, None)
        raise e

if __name__=="__main__":
    main()
