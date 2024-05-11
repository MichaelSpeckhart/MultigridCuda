
__global__ void initMatrixA(double* A, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int totalSize = N * N;
    if (idx < totalSize) {
        int i = idx / N;
        int j = idx % N;
        A[idx] = 0.0;
        if (i == j) A[idx] = 2.0;
        if (i > 0 && j == i - 1) A[idx] = -1.0;
        if (i < N - 1 && j == i + 1) A[idx] = -1.0;
    }
}

__global__ void initVectors(double* x, double* b, int N, double fillValue) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        x[idx] = 0.0;
        b[idx] = fillValue;
    }
}

__global__ void matrixVectorProduct(double* A, double *x, double *y, int numRows, int numCols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows) {
        double sum = 0.0;
        for (int col = 0; col < numCols; col++) {
            sum += A[row * numCols + col] * x[col];
        }
        y[row] = sum;
    }
}

__global__ void redBlackGaussSeidel(double *A, double *x, double *b, int N, int color) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (i < N && j < N) {
        int idx = i * N + j; // Linear index in the matrix/vector
        // Check if it's the correct phase (color) to update
        if ((i + j) % 2 == color) {
            double sigma = 0.0;
            for (int k = 0; k < N; ++k) {
                if (k != j) { // Exclude the diagonal element
                    sigma += A[idx * N + k] * x[k];
                }
            }
            x[j] = (b[idx] - sigma) / A[idx * N + j]; // Update the current point
        }
    }
}

#include <cstdio>


int main() {
    int N = 1000;
    double *A;
    double *x;
    double *b;

    cudaMalloc(&A, N * N * sizeof(double));
    cudaMalloc(&x, N * sizeof(double));
    cudaMalloc(&b, N * sizeof(double));

    double fillValue = 1.0;
    int blockSize = 256;
    int numBlocks = (N * N + blockSize - 1) / blockSize;
    initMatrixA<<<numBlocks, blockSize>>>(A, N);

    

    numBlocks = (N + blockSize - 1) / blockSize;
    initVectors<<<numBlocks, blockSize>>>(x, b, N, fillValue);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record event on the default stream (stream 0) before kernel launch
    cudaEventRecord(start);

    // int maxIters = 100;
    // double tol = 1e-6;
    // int iter = 1;
    // while ((iter < maxIters)) {
    //     redBlackGaussSeidel<<<numBlocks, blockSize>>>(A, x, b, N, 0);
    //     cudaDeviceSynchronize();
    //     redBlackGaussSeidel<<<numBlocks, blockSize>>>(A, x, b, N, 0);
    //     cudaDeviceSynchronize();

    //     iter++;
    // }

    matrixVectorProduct<<<numBlocks, blockSize>>>(A, x, b, N, N);

    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %f milliseconds\n", milliseconds);
}