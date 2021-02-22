#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define M 3

int main (void){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    float* h_a = 0;     // Host array a
    float* d_a;         // Device array a
    float* h_b = 0;     // Host array b
    float *d_b;         // Device array b
    float result = 0;   // Result

    h_a = (float *)malloc (M * sizeof (*h_a));      // Create memory for h_a and initialize
    if (!h_a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    h_a[0] = 3.0;
    h_a[1] = 10.0;
    h_a[2] = 20.0;
    
    h_b = (float *)malloc (M * sizeof (*h_b));  // Create memory for h_b and initialize
    if (!h_b) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    h_b[0] = 5.0;
    h_b[1] = 10.0;
    h_b[2] = 15.0;
    
    cudaStat = cudaMalloc ((void**)&d_a, M*sizeof(*h_a));       // Create memory for d_a
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    
    cudaStat = cudaMalloc ((void**)&d_b, M*sizeof(*h_b));       // Create memory for d_b
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    
    stat = cublasCreate(&handle);               // Create handler
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    
    stat = cublasSetVector(M,sizeof(float),h_a,1,d_a,1);    // h_a -> d_a
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (d_a);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    stat = cublasSetVector(M,sizeof(float),h_b,1,d_b,1);    // h_b -> d_b
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (d_b);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    stat = cublasSdot(handle,M,d_a,1,d_b,1,&result);        // Dot product function
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed cublasSdot");
        cudaFree (d_a);
        cudaFree (d_b);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    printf("Result of dot product --> %f",result);
    
    cudaFree (d_a);     // Deallocate d_a
    cudaFree (d_b);     // Deallocate d_b
    
    cublasDestroy(handle);  // Destroy handle
    
    free(h_a);      // Deallocate h_a
    free(h_b);      // Deallocate h_b    
    return EXIT_SUCCESS;
}
