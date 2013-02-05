//#include <WINDOWS.H>      
#include <stdio.h>
#include <math.h>
//#include <malloc.h>

//#define INF 1.0e99
//#define EPS 1.0e-14
#define E  2.7182818284590452353602874713526625
#define PI 3.1415926535897932384626433832795029


int square(int x, int f)
{
  f=x*x;
  return f;
}


double sinus(double x)
{
    return sin(x);
}

double sunis(double x)
{
    return sin(1./x);
}

double sum(double *x,int n)
{
  int i;
  double counter;
  counter = 0;
  for(i=0;i<n;i++)
    {
      counter=counter+x[i];

    }
  return counter;
}

float add(float f1, float f2)
{
    return f1 + f2;
}

float avg_array(float *array, int num_entries)
{
    int i;
    float sum = 0.0f;

    for (i = 0; i < num_entries; i++)
    {
        sum += array[i];
    }
    return sum / num_entries;
}