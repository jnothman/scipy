from functools import wraps

import numpy as np


class _delegate(object):
    def __init__(self, attr):
        self.attr = attr

    def __get__(self, obj, objtype=None):
        return getattr(obj.negative, self.attr)

    def __set__(self, obj, value):
        return setattr(obj.negative, self.attr, value)


class _delegate_inverted(object):
    # handles multiple datatypes
    def __init__(self, attr):
        self.attr = attr

    def __get__(self, obj, objtype=None):
        delegate = getattr(obj.negative, self.attr)
        @wraps(delegate)
        def fn(*args, **kwargs):
            return ~delegate(*args, **kwargs)
        return fn


class _delegate_mtm_nocopy(object):
    def __init__(self, attr):
        self.attr = attr

    def __get__(self, obj, objtype=None):
        delegate = getattr(obj.negative, self.attr)
        @wraps(delegate)
        def fn(*args, **kwargs):
            return mostly_true_matrix(delegate(*args, **kwargs), copy=False)
        return fn


class mostly_true_matrix(object):
    """Wraps a sparse data matrix to represent its boolean negation
    """

    def __init__(self, negative, copy=True):
        # TODO: validation, etc.
        if negative.dtype != np.bool:
            negative = negative.astype(bool)
        elif copy:
            negative = negative.copy()
        self.negative = negative

    def __invert__(self):
        return self.negative.copy()

    def __repr__(self):
        return '<Boolean inversion of {!r}>'.format(self.negative)

    def __str__(self):
        return (str(self.negative)
            .replace('True', 'eslaF')
            .replace('False', 'True')
            .replace('eslaF', 'False'))

    shape = _delegate('shape')
    reshape = _delegate('shape')
    dtype = _delegate('dtype')  # XXX: constrain?
    __len__ = _delegate('__len__')
    size = _delegate('size')
    format = _delegate('format')
    asformat = _delegate_mtm_nocopy('asformat')
    tobsr = _delegate_mtm_nocopy('tobsr')
    tocoo = _delegate_mtm_nocopy('tocoo')
    tocsc = _delegate_mtm_nocopy('tocsc')
    tocsr = _delegate_mtm_nocopy('tocsr')
    todia = _delegate_mtm_nocopy('todia')
    todok = _delegate_mtm_nocopy('todok')
    tolil = _delegate_mtm_nocopy('tolil')
    copy = _delegate_mtm_nocopy('copy')
    transpose = _delegate_mtm_nocopy('transpose')
    __getitem__ = _delegate_inverted('__getitem__')
    getrow = _delegate_mtm_nocopy('getrow')
    getcol = _delegate_mtm_nocopy('getcol')
    diagonal = _delegate_inverted('diagonal')
    toarray = _delegate_inverted('toarray')

    @property
    def T(self):
        return self.transpose()

    def __iter__(self):
        for r in self.negative:
            # XXX: is copy necessary?
            yield mostly_true_matrix(r)

    # TODO: __nonzero__?

    def sum(self, axis=None):
        res = self.negative.sum(axis)
        if axis is None:
            return np.prod(self.shape) - res
        if axis == 0:
            return self.shape[0] - res
        return self.shape[1] - res

    @property
    def __setitem__(self):
        delegate = self.negative.__setitem__
        @wraps(delegate)
        def fn(key, val):
            delegate(key, np.invert(val))
        return fn
