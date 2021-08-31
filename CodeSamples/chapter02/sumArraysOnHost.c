#include <stdlib.h>
#include <time.h>

/*
 * This example demonstrates a simple vector sum on the host. sumArraysOnHost
 * sequentially iterates through vector elements on the host.
 */

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }

}

void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }

    return;
}

int main(int argc, char **argv)
{
    //1.首先分配内存
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    //2.然后根据首地址和长度，初始化h_A和h_B
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    //3.然后加就可以了。
    sumArraysOnHost(h_A, h_B, h_C, nElem);

    free(h_A);
    free(h_B);
    free(h_C);

    return(0);
}
