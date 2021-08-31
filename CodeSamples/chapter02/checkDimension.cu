#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * Display the dimensionality of a thread block and grid from the host and
 * device.
 */
//
//对于每一个线程而言，blockDim都是一样的，因为blockDim表示这个线程所在的这个线程块的结构。
//threadIdx则表示这个线程在这个线程块中的位置，所以对于每个线程都不一样（从0开始。）
//gridDim表示这个线程所在的网格的整体结构
//blockIdx表示这个线程所在的线程块在网格中的位置。
//所以说我们会发现由于网格中有多个线程块，单凭threadIdx是无法唯一表示一个线程的。
//只有threadIdx+blockIdx才能唯一表示一个线程，这点还是挺复杂的。
__global__ void checkIndex(void)
{
    printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

    printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

}

int main(int argc, char **argv)
{
    // define total data element
    int nElem = 6;

    // define grid and block structure
    //每个线程块有3个线程。
    dim3 block(3);
    //一共有8/3=2个线程块。
    dim3 grid((nElem + block.x - 1) / block.x);

    // check grid and block dimension from host side
    //看来host来看的话，y,z是默认都初始化为1。
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    // check grid and block dimension from device side
    checkIndex<<<grid, block>>>();

    // reset device before you leave
    CHECK(cudaDeviceReset());

    return(0);
}
