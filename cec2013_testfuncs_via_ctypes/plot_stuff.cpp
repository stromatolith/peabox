#include <WINDOWS.H>    
#include <stdio.h>
#include <math.h>
#include <malloc.h>

#include <iostream>
using namespace std;

void test_func(double *, double *,int,int,int);

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag;

int main()
{
    int i,j,k,n,m,h,func_num;
    double xc, yc, xlim, ylim, xwidth, ywidth, dx, dy;
    double *f,*x;
    FILE *fpt;
    
    m=201;
    h=181;
    n=2;
    func_num=9;
    xlim=50.; ylim=50.; xwidth=2*xlim; ywidth=2*ylim;
    dx=xwidth/(m-1); dy=ywidth/(h-1);

    fpt=fopen("output/test_data.txt","w");
    if (fpt==NULL)
    {
        printf("\n Error: Cannot open input file for reading \n");
    }
    x=(double *)malloc(m*n*sizeof(double));
    f=(double *)malloc(sizeof(double)  *  m);

    for (i = 1; i < h; i++)
    {
        yc=ylim-i*dy;
        for (j = 0; j < m; j++)
        {
            xc=-xlim+j*dx;
            x[j*n]=xc;
            x[j*n+1]=yc;
        }
        printf("first DNA: %f, %f\n",x[0],x[1]);
        printf("2nd DNA: %f, %f\n",x[2],x[3]);
        printf("last DNA: %f, %f\n\n",x[2*m-2],x[2*m-1]);
        test_func(x, f, n,m,func_num);

        for (j = 0; j < m; j++)
        {
            fprintf(fpt,"%e   ",f[j]);
        }
        fprintf(fpt,"\r\n");
    }

    

    
    fclose(fpt);
    free(x);
    free(f);
    free(y);
    free(z);
    free(M);
    free(OShift);
    free(x_bound);    
    
    return 0;

}

