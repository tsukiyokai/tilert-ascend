import cython


cdef class _Node:
    cdef public Py_ssize_t hi, length, latest
    cdef public dict children


cdef class LookupDrafter:
    cdef dict __dict__
    cdef object __weakref__
    cdef public list context
    cdef public object longest, shortest, hits, rounds
    cdef public _Node _root

    @cython.locals(length=Py_ssize_t, hi=Py_ssize_t, context=list,
                   offset=Py_ssize_t)
    cdef Py_ssize_t _shared(self, _Node child, Py_ssize_t query_hi,
                           Py_ssize_t remaining) except -1

    @cython.locals(window_length=Py_ssize_t, node=_Node, depth=Py_ssize_t,
                   context=list, query_hi=Py_ssize_t, child=_Node,
                   shared=Py_ssize_t, replacement=_Node, token=object)
    cdef void _insert(self, Py_ssize_t end) except *

    @cython.locals(maximum=Py_ssize_t, node=_Node, matched=Py_ssize_t,
                   end=Py_ssize_t, context=object, query_hi=Py_ssize_t,
                   child=_Node, shared=Py_ssize_t, proposed=list)
    cpdef list propose(self, object count)

    cpdef dict report(self)
