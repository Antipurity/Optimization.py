"""
Stochastic optimization of arbitrary self-modifying code by an arbitrary measure.
A curious thing: just adding random numbers to static variables under a goal would automatically improve those numbers.
Best used with repeated computations, in a mostly virtualized or otherwise unimportant environment — not once and on precious things.

`@static(…)` annotation is used for variables that can self-modify.
`read`/`write`/`writes`/`commit` are used for delaying commitment of modifications if needed.
`goal(measure, func, …)` and `on_goal(do)` are used to specify a goal and a way of reaching it (an optimizer registered from inside of `func`).
`minimize(measure, …)`/`maximize(measure, …)` optimizes by repeating computation and picking and commiting the best result.


Examples:


@static(a=1, b=2)
def f1(static):
    static.a += random.uniform(-1,1)
    static.b += random.uniform(-1,1)
    return static.a + static.b
def m1():
    return minimize(lambda x: x*x-5*x, f1, tries=100)
print(m1()) # Prints something close to 2.5.


@static(a = Prob(Bounded(0.5, 1)))
def f2(static):
    return static.a
print( sum(1 if f2() else 0 for i in range(10000)) ) # Prints about 7500.


@static(a = Bounded(0, 1, Walk(0.5, Bounded(-.2, .2))))
def f3(static):
    return static.a
for i in range(100):
    print(round(f3(), 2)) # Prints numbers in 0…1, changing gradually.
"""

import random
import math

def __checkOverride(data, code, *args, **kwargs):
    override = None
    try:
        override = data.__concept__[code]
    except AttributeError:
        return None
    if override is not None:
        return override(*args, **kwargs)
    return None

def checked(func):
    """Makes `func` check its overrides in arguments, returning the result of that instead of `func` if present."""
    def check(*args, **kwargs):
        for v in args:
            result = __checkOverride(v, check, *args, **kwargs)
            if result is not None:
                return result
        for k in kwargs:
            result = __checkOverride(kwargs[k], check, *args, **kwargs)
            if result is not None:
                return result
        return func(*args, **kwargs)
    check.__annotations__ = func.__annotations__
    return check

def concept(view):
    """
    @concept({ f:becomes }):
    A way to specify overrides of `checked` functions.
    """
    def setConcept(on):
        on.__concept__ = view
        return on
    return setConcept


def _dictSet(dict, key, value):
    """
    Makes sure that even unhashable keys can be stored in a dict.
    """
    try:
        dict[key] = value
    except TypeError:
        # key is not hashable.
        if id(key) not in dict:
            if _dictKeys not in dict: dict[_dictKeys] = []
            dict[_dictKeys].append(key)
        dict[id(key)] = value
def _dictCreate(dict, key):
    """
    When _dictSet needs to go through some hierarchy, creating dicts if missing, this is the function to call for dicts on the way.
    """
    if id(key) in dict:
        return dict[id(key)]
    else:
        try:
            if key in dict: return dict[key]
        except TypeError: pass
        d = {}
        _dictSet(dict, key, d)
        return d
def _dictGet(dict, key):
    """
    A counterpart to _dictSet, gets keys properly from a dict.
    """
    if id(key) in dict:
        return dict[id(key)]
    else:
        try: return dict[key]
        except TypeError: raise KeyError(key)
def _dictKeys(dict):
    """
    A counterpart to _dictSet, returns an iterator of all keys.
    """
    if _dictKeys in dict:
        for k in dict[_dictKeys]: yield k
        ids = set(id(k) for k in dict[_dictKeys])
        for k in dict.keys():
            if k is not _dictKeys:
                if k not in ids:
                    yield k
    else:
        for k in dict.keys():
            if k is not _dictKeys:
                yield k

def read(dict, key=None):
    """
    Reads the value under a key from a dictionary; `key` should be None to treat `dict` as the read value. The value can override `read(value)`. Knows about `writes`'s virtualization.
    """
    # See if what we want is in writes, then if it's in reads, then perform the actual read (allowing the value to override it).
    if commit.writes is not None:
        try:
            if key is None: return _dictGet(commit.writes, dict)
            else: return _dictGet(_dictGet(commit.writes, dict), key)
        except KeyError: pass
    if commit.reads is not None:
        try:
            if key is None: return _dictGet(commit.reads, dict)
            else: return _dictGet(_dictGet(commit.reads, dict), key)
        except KeyError: pass
    try:
        value = dict[key] if key is not None else dict
        result = __checkOverride(value, read, value)
        if result is not None:
            if key is not None and commit.reads is not None:
                if key is None: _dictSet(commit.reads, dict, result)
                else: _dictSet(_dictCreate(commit.reads, dict), key, result)
            return result
        return value
    except KeyError:
        pass

def write(dict, key=None, new=None):
    """
    Writes a value to a key in a dictionary and returns the new value; `key` should be None to treat `dict` as the written-to value. The previous value can override `write(prev, new)` to become its result. Virtualized (deferred until `commit`) if done inside `journal`.
    """
    if commit.writes is not None:
        if key is None: return _dictSet(commit.writes, dict, new)
        else: return _dictSet(_dictCreate(commit.writes, dict), key, new)
    else:
        try:
            prev = dict[key] if key is not None else dict
            # (Virtualizing the actual reads in the journal too (and caching them) would allow us to know what value was actually read, and allow custom-objects to react to that. Choice has no choice but to re-read its condition without this.)
            result = __checkOverride(prev, write, prev, new)
            if result is not None:
                if key is not None: dict[key] = result
                return result
            else:
                if key is not None: dict[key] = new
                return new
        except KeyError:
            if key is not None: dict[key] = new
            return new

def journal(func, *args, into = None, **kwargs):
    """
    Collects reads and writes that `func` performs during its call. Returns (result, read_journal, write_journal).
    """
    (prev_reads, prev_writes) = (commit.reads, commit.writes)
    r = commit.reads = {}
    w = commit.writes = {}
    w[_dictKeys] = []
    result = func(*args, **kwargs)
    (commit.reads, commit.writes) = (prev_reads, prev_writes)
    return (result, r, w)

def commit(result_and_write_journal, dont = False):
    """
    Actually performs the virtualized writes. Takes (result, read_journal, write_journal); returns result.
    Pass in dont=True to just return result, so we don't have another function for just that.
    """
    if not dont:
        r = result_and_write_journal[1] # Ignore this.
        w = result_and_write_journal[2]
        for dict in _dictKeys(w):
            dict.update(_dictGet(w, dict))
    return result_and_write_journal[0]
commit.reads = None
commit.writes = None




class SelfWritingDict:
    """
    Wrappers for dictionaries that use `read` and `write` to read/write, and also immediately write back the read value, and also using attributes instead of items (so `dict.a` instead of `dict['a']`).
    """
    __slots__ = 'dict'
    def __init__(self, dict = {}):
        object.__setattr__(self, 'dict', dict)
    def __getattr__(self, key):
        value = read(self.dict, key)
        write(self.dict, key, value)
        return value
    def __setattr__(self, key, value):
        if self.dict is not None:
            write(self.dict, key, value)
    def __delattr__(self, key):
        del self.dict[key]
    def __dir__(self):
        return self.dict.keys()

def static(**kwargs):
    """
    @static(a=1, b=2):
    An annotation; the function will be able to access a static storage through its first argument.
    Creates a SelfWritingDict of the keyword-args supplied initially. Amenable to `goal` optimization.
    """

    dict = SelfWritingDict(kwargs)
    # Creating a class object with the exact __slots__ we need would probably be more performant than using a dictionary.
    def funcModifier(func):
        return lambda *args, **kwargs: func(dict, *args, **kwargs)
    return funcModifier



def goal(measure, func, *args, **kwargs):
    """
    Calls func and returns its result, likely optimized via any `on_goal` registered inside.
    `measure` should be a function accepting results of `func` and returning a number (or something that can be compared and inverted).
    """
    (prev_do, prev_in_goal) = (goal.do, goal.in_goal)
    (goal.do, goal.in_goal) = (None, True)
    goal.journaled = journal(func, *args, **kwargs)
    if goal.do is not None:
        prev_in_do = goal.in_do
        goal.in_do = True
        result = goal.do(measure, func, *args, **kwargs)
        goal.in_do = prev_in_do
        (goal.do, goal.in_goal) = (prev_do, prev_in_goal)
        return result
    (goal.do, goal.in_goal) = (prev_do, prev_in_goal)
    return commit(goal.journaled)
goal.journaled = None
goal.do = None
goal.in_do = False
goal.in_goal = False

def on_goal(do):
    """
    Specifies an optimizer: when `goal` returns, `do` will be called with `goal`'s arguments.

    `goal.journaled` holds the original result and its journal (holds the result of `journal`).
    If several `on_goal`s are called, the first registered optimizer always goes first, and its `func` will call the next one. This is so that optimization methods can be stacked within each other, with the first being the most important (if they're limited by execution time).
    Calling this within a goal optimizer has no effect, so that optimizers can call `func` several times and return the best result.
    Calling this not within a goal has no effect.
    Does not handle async results (since that would require effort, as there is no automatic async-rewriting).
    """
    if not goal.in_goal:
        return
    if goal.in_do:
        return
    if goal.do is None:
        goal.do = do
    else:
        prev = goal.do
        def combine(measure, func, *args, **kwargs):
            def redo(*args, **kwargs):
                return do(measure, func, *args, **kwargs)
            return prev(measure, redo, *args, **kwargs)
        goal.do = combine

def minimize(measure, func = None, *args, tries=2, **kwargs):
    """
    Minimizes a computation's result by repeating it and picking and commiting the min-measure one. Optimization of arbitrary self-modifying code.
    """
    if func is None:
        def repeat(measure, func, *args, **kwargs):
            (bestJ, bestM) = (goal.journaled, measure(commit(goal.journaled, dont=True)))
            for i in range(tries):
                j = journal(func, *args, **kwargs)
                m = measure(commit(j, dont=True))
                if m < bestM:
                    (bestJ, bestM) = (j, m)
            return commit(bestJ)
        on_goal(repeat)
    else:
        def immediately_minimize(*args, **kwargs):
            minimize(measure, tries=tries)
            return func(*args, **kwargs)
        return goal(measure, immediately_minimize, *args, **kwargs)

def maximize(measure, func = None, *args, tries=2, **kwargs):
    """
    Maximizes a computation's result by repeating it and picking and commiting the max-measure one. Optimization of arbitrary self-modifying code.
    The opposite of `minimize`.
    """
    return minimize(lambda r: -measure(r), func, *args, tries=tries, **kwargs)





class Bounded:
    """
    A float number from `a` to `b`, possibly with the read value `v`.
    On read, returns clamped `v` if specified or a random.uniform number from `a` to `b`. On write, writes to `v` or does nothing.
    """
    def __init__(self, a, b, v = None):
        if (b < a):
            raise Exception('Invalid ordering of bounds on a number')
        self.a = a
        self.b = b
        self.v = v
    def _boundedRead(self):
        if self.v is None:
            return random.uniform(self.a, self.b)
        v = read(self.v)
        return self.a if v < self.a else self.b if self.b < v else v
    def _boundedWrite(self, value):
        if self.v is not None:
            self.v = write(self.v, new=value)
        return self
    __concept__ = { read: _boundedRead, write: _boundedWrite }

class Walk:
    """
    Represents a number that has another number added to it on each write.
    """
    __slots__ = ('v', 'd')
    def __init__(self, value, delta):
        self.v = value
        self.d = delta
        pass
    def _boundedRead(self):
        return read(self.v) + read(self.d)
    def _boundedWrite(self, value):
        self.v = write(self.v, new = value)
        return self
    __concept__ = { read: _boundedRead, write: _boundedWrite }

class Prob:
    """
    Represents a boolean with known-probability values.
    On read, returns `True` with probability `p`, `False` otherwise. On write, writes `1 if value else 0` to `p`.
    """
    __slots__ = 'p'
    def __init__(self, p):
        self.p = p
    def _boundedRead(self):
        return True if random.uniform(0,1) < read(self.p) else False
    def _boundedWrite(self, value):
        if self.p is not None:
            self.p = write(self.p, new = 1 if value else 0)
        return self
    __concept__ = { read: _boundedRead, write: _boundedWrite }

class If:
    """
    Represents `then` if `condition` (a boolean) else `Else`.
    `condition` is checked for both reading and writingwriting (returning the same thing thanks to caching reads).
    """
    __slots__ = ('c', 'a', 'b')
    def __init__(self, condition, then, Else):
        self.c = condition
        self.a = then
        self.b = Else
    def _boundedRead(self):
        return read(self.a) if read(self.c) else read(self.b)
    def _boundedWrite(self, value):
        if read(self.c):
            self.a = write(self.a, new = value)
        else:
            self.b = write(self.b, new = value)
        return self
    __concept__ = { read: _boundedRead, write: _boundedWrite }







if __name__ == '__main__':
    @static(a = Bounded(0, 1, Walk(0.5, Bounded(-.2, .2))))
    def f3(static):
        return static.a
    for i in range(100):
        print(round(f3(), 2)) # Prints numbers in 0…1, changing gradually.
