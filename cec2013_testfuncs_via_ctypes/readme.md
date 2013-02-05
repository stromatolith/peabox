How to call the CEC-2013 test functions using ctypes
----------------------------------------------------

The CEC-2013 is a conference on evolutionary computation, and they have a competition
for evolutionary algorithms each year.
link for the conference: http://www.cec2013.org/
link for the competitions: http://www.cec2013.org/?q=Competitions

As I do all my coding in python, I needed to get the test functions to work in a python program,
which I managed using ctypes, a module of the Python Standard Library for incorporating
shared libraries written in C. For me as an absolute C-newbie this was kinda tricky, I had to
google a lot and go through some starter tutorials for the language C and python's ctypes module.
Here is the necessary code to make it happen.

### step 1:
- download the C-files from the link they give here: http://www.ntu.edu.sg/home/EPNSugan/index_files/CEC2013/CEC2013.htm

### eventual step 1.5:
- do step 2 first with a little test C-library, like myfuncs.c and use_myfuncs_dll.py

### step 2:
- modify the file test_func.cpp so it is named test_func.c (I don't know if it is necessary, but as it seems to be pure C code
and it is to be compiled as such, I think it makes sense to clarify.)
- copy these header lines from main.cpp to test_func.c:

~~~
void test_func(double *, double *,int,int,int);
double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag;
~~~

- now these lines can be commente out

~~~
//extern double *OShift,*M,*y,*z,*x_bound;;
//extern int ini_flag,n_flag,func_flag;
~~~


### step 3, compiling the dynamic library (Linux, Kubuntu to be more exact):
- compile the file with `gcc -c test_func.c`
- finally create a shared library from it with `gcc -shared -o test_func.so test_func.o`

### step 3 (as it didn't work for me), compiling the dynamic library (Windows+Cygwin):
- in the command prompt window:
- compile the file with `gcc -c test_func.c`
- finally create a shared library from it with `gcc -shared -o test_func.dll test_func.o`
Step 3 so far was no problem, but when using the dll in python I got my fitness array filled with wrong values.


step 4:
- the script plot_testfunc.py gives an example of how to use the library in python
- create some plots

step 5:
for verification that that the plots are not just nice, but also give the correct landscapes:
- use the C-program plot_stuff.cpp to get data for plots into a file (this one I compiled in the netbeans IDE and there
I didn't care about whether C or C++ ... just whether it worked, sorry)
- use the script plot_data.py to make a comparison plot from that output data file



here the links I found useful during that tinkering:
http://jjd-comp.che.wisc.edu/index.php/PythonCtypesTutorial
http://cygwin.com/cygwin-ug-net/dll.html
http://eli.thegreenplace.net/2008/08/31/ctypes-calling-cc-code-from-python/
http://www.scipy.org/Cookbook/Ctypes
