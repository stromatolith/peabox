#!python
"""
I want to learn how to complile C-code into a dll and use it from within python
this here was the first thing I got working, I used these two sources:
http://eli.thegreenplace.net/2008/08/31/ctypes-calling-cc-code-from-python/
http://cygwin.com/cygwin-ug-net/dll.html
"""
from numpy import pi
from ctypes import cdll, c_double  #, c_long, c_int, c_char_p, create_string_buffer
import ctypes as ct

#mf = cdll.myfuncs    # for using myfuncs.dll under windows
mf = cdll.LoadLibrary('./myfuncs.so')    # Linux

a=c_double()

mf.square.argtypes = [ct.c_int,ct.c_int]
mf.sinus.argtypes = [ct.c_double]
mf.sunis.argtypes = [ct.c_float]
mf.add.argtypes = [ct.c_float,ct.c_float]
mf.avg_array.argtypes = [ct.POINTER(ct.c_float), ct.c_int]
mf.sum.argtypes = [ct.POINTER(ct.c_double), ct.c_int]


mf.sinus.restype = ct.c_double
mf.add.restype = ct.c_float
mf.sum.restype = ct.c_double
mf.avg_array.restype = ct.c_float

a=ct.c_float(2.0)
b=ct.c_float(3.44)
c=ct.c_float(1.7)
d=ct.c_double(0.499*pi)

datlist = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
dat = (ct.c_double * len(datlist))()
for i,val in enumerate(datlist):
    dat[i] = val

s1=mf.add(a,b)
print 'mf.add for ',a.value,b.value,'  yields ',s1
r1=mf.sinus(d)
print 'mf.sinus for ',d.value,'  yields ',r1
s2=mf.sum(dat,ct.c_int(len(datlist)))
print 'mf.sum for ',[dat[i] for i in range(len(datlist))],'  yields ',s2

