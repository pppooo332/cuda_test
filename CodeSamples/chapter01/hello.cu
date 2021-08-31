#include "../common/common.h"
#include <stdio.h>
#include <unistd.h>

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */
//修饰符__global__告诉编译器这个函数将会从CPU中调用，然后在GPU上执行。
__global__ void helloFromGPU()
{
    printf("Hello World from GPU!\n ");
}

int main(int argc, char **argv)
{
    printf("Hello World from CPU!\n thread %d",threadIdx.x);
    //10个线程执行这个函数
    helloFromGPU<<<1, 10>>>();
    //用来显式地释放和清空当前进程中与当前设备有关的所有资源。
    //注释掉这个之后，GPU执行不了print，难道是因为这个函数也有阻塞的功能，让主线程不要过早结束？
    //果然，加上sleep之后就可以输出了。
    //sleep(2);
    CHECK(cudaDeviceReset());
    return 0;
}
//nvcc -o hello hello.cu
//./hello

