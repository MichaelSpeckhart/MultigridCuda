

__device__ double p_norm(double* a, double p, size_t n) {
    double norm = 0.0;
    for (size_t i = 0; i < n; ++i) {
        norm += pow(abs(a[i]), p);
    }

    return pow(norm, 1 / p);
}

__device__ double l2_norm( double* a, size_t n) {
    double norm = 0.0;
    for (size_t i = 0; i < n; ++i) {
        norm += a[i] * a[i];
    }

    return sqrt(norm);
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


#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

const int X = 5;
const double d = 1.0;
const double c = 0.0;
const double bs = 0.0;

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

__global__ void weightedJacobi(double* A, double *b, double *x, double *xNew, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double omega = 2.0 / 3.0;  // Correct division to floating-point
    if (i < N) {
        double sigma = 0.0;
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                sigma += A[i * N + j] * x[j];
            }
        }
        xNew[i] = omega * (b[i] - sigma) / A[i * N + i] + (1 - omega) * x[i];
    }
}

__global__ void weightedJacobiLast(double* A, double *b, double *x, double *xNew, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double omega = 2.0 / 3.0;  // Correct division to floating-point
    if (i < N) {
        double sigma = 0.0;
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                sigma += A[i * N + j] * x[j];
            }
        }
        xNew[i] = omega * (b[i] - sigma) / A[i * N + i] + (1 - omega) * x[i];
    }
    // No pointer assignment, as it doesn't work as intended
}

__global__ void weightedJacobiFirst(double* A, double *b, double *x, double *xNew, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double omega = 2.0 / 3.0;  // Correct division to floating-point
    if (i < N) {
        double sigma = 0.0;
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                sigma += A[i * N + j] * x[j];
            }
        }
        xNew[i] = omega * (b[i] - sigma) / A[i * N + i] + (1 - omega) * x[i];

        
    }
    __syncthreads();
}


__global__ void calculateResidual(double *A, double *x, double *b, double *residual, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        double Ax = 0.0;
        for (int j = 0; j < N; ++j) {
            Ax += A[i * N + j] * x[j];
        }
        residual[i] = b[i] - Ax;
    }
}

__global__ void fullWeightRestriction(double* f2h, double *residual, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int nh = N / 2;  // Half the size of N, assuming N is even.
    if (i < nh) {
        if (2 * i + 2 < N) { // Check to avoid out-of-bounds access
            f2h[i] = (residual[2 * i] + 2 * residual[2 * i + 1] + residual[2 * i + 2]) / 4.0;
        }
    }
}

__global__ void errorCorrection(double* x2h, double *e2h, int N) {
    for (int i = 0; i < N; ++i) {
        x2h[i] += e2h[i];
    }
}

__global__ void residualCorrection(double* r2h, double *f2h, double *Ax2h, int N) {
    for (int i = 0; i < N; ++i) {
        r2h[i] = f2h[i] - Ax2h[i];
    }
}

__global__ void errorProlongation(double *eh, double *x2h, int N) {
    int n2h = N / 2;
    // Assign values from x2h to every second element of eh, starting at index 1 (0-based)
    for (int i = 1; i < N - 1 && (i / 2) < n2h; i += 2) {
        eh[i] = x2h[i / 2];
    }

    // Interpolate to find values at odd indices based on the new values at even indices
    for (int i = 0; i < N - 2; i += 2) {
        if (i + 1 < N) eh[i + 1] = 0.5 * eh[i + 2]; // Average with the next even-indexed element
    }

    // Adjust values at odd indices again by adding half the previous even-indexed values
    for (int i = 3; i < N; i += 2) {
        if (i - 1 < N) eh[i] += 0.5 * eh[i - 1]; // Add half of the previous even-indexed value
    }

    // for (int i = 0; i < N / 2; ++i) {
    //     printf("X2[%d] = (%f)\n", i, x2h[i]);
    // }
}

__global__ void checkInitialization(double *array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        printf("Array[%d] = %f\n", idx, array[idx]);
    }

}

std::vector<double> solveJacobi(double *d_A, double *d_x, double *d_b, double *d_xNew, int N) {
    // Initial guess for x (could be zero or another guess)
    std::vector<double> h_x(N, 0.0);  // Initial guess on the host
    cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    double tol = 1e-6;
    double diff = 1.0;
    int max_iter = 100;
    int iter = 0;

    std::vector<double> h_xNew(N);

    while (iter < max_iter && diff > tol) {
        weightedJacobiFirst<<<1, 1>>>(d_A, d_b, d_x, d_xNew, N);
        cudaDeviceSynchronize();

        // Copy xNew back to host to check for convergence
        cudaMemcpy(h_xNew.data(), d_xNew, N * sizeof(double), cudaMemcpyDeviceToHost);

        diff = 0.0;
        for (int i = 0; i < N; i++) {
            double abs_diff = fabs(h_xNew[i] - h_x[i]);
            if (abs_diff > diff) {
                diff = abs_diff;
            }
        }

        if (diff < tol) {
            std::cout << "Converged!!!\n";
            break;
        }

        // Update h_x to the latest h_xNew for the next iteration's comparison
        h_x = h_xNew;

        // Swap pointers to avoid copying data
        double *temp = d_x;
        d_x = d_xNew;
        d_xNew = temp;

        iter++;
    }

    // Copy the final result back to host
    cudaMemcpy(h_x.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost);


    return h_x;
}

__global__ void discretizeGrid(double *A2h, int N, double h, double hs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < N; ++i) {
        double val_left = -((1.0 / hs) + (0.5 * bs / hs)) * d;
        double val_mid = ((2.0 / hs) + c) * d;
        double val_right = -((1.0 / hs) - (0.5 * bs / hs)) * d;

        int index = i * N + i;  
        A2h[index] = val_mid;
        if (i > 0) {
            A2h[index - 1] = val_left;  
        }
        if (i < N - 1) {
            A2h[index + 1] = val_right;  
        }
    }
    // printf("A2h contents\n");
    // for (int i = 0; i < N; ++i) {
    //     printf("A2[%d] = (%f)\n", i, A2h[i]);
    // }
    // printf("A2h contents done\n");
}

std::vector<double> coarseGridCorrection(double *A, double *r, double tol, int N) {
    int n2h = ceil((N) / 2);
    printf("Size of A2h = %d\n", n2h);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Mallocing what we need, going from finer to coarser grid
    double *d_x2, *d_f2, *d_A2, *d_e2;
    cudaMalloc(&d_x2, n2h* sizeof(double));
    cudaMalloc(&d_f2, n2h * sizeof(double));
    cudaMalloc(&d_A2, (n2h) * (n2h) * sizeof(double));
    cudaMalloc(&d_e2, n2h * sizeof(double));

    std::vector<double> h_f2;
    std::vector<double> h_r(N, 0.0);
    cudaMemcpy(h_r.data(), r, N * sizeof(double), cudaMemcpyDeviceToHost);

    // printf("Residual in host memory\n");
    // for (int i = 0; i < h_r.size(); ++i) {
    //     printf("Host res[%d] = %f\n", i, h_r[i]);
    // }

    for (int i = 1; i < N; i += 2) {
        if (i + 1 < n2h - 1) {
            double value = h_r[i-1] + (2.0 * h_r[i]) + (h_r[i+1]);
            h_f2.push_back(h_r[i-1] + (2.0 * h_r[i]) + (h_r[i+1]));
        }
        else {
            double value = h_r[i-1] + (2.0 * h_r[i]);
            h_f2.push_back(h_r[i-1] + (2.0 * h_r[i]));
        }
    }

    // printf("F2H after full weight restriction\n");
    // for (int i = 0; i < n2h; ++i) {
    //     printf("Host F2[%d] = %f\n", i, h_f2[i]);
    // }

    cudaMemcpy(d_f2, h_f2.data(), n2h * sizeof(double), cudaMemcpyHostToDevice);

    // printf("Residual in Vec\n");
    // for (int i = 0; i < h_r.size(); ++i) {
    //     printf("Residual in vec[%d] = (%f)\n", i, h_r[i]);
    // }

    // fullWeightRestriction<<<numBlocks, blockSize>>>(d_f2, r, N);
    // cudaDeviceSynchronize();

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    // }

    // std::vector<double> h_f2(n2h, 0.0);
    // cudaMemcpy(h_f2.data(), d_f2, N * sizeof(double), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < h_f2.size(); ++i) {
    //     printf("F2[%d] = (%f)\n", i, h_f2[i]);
    // }

    // std::cout << "Restriced Residual after Jacobi Iteration\n";
    // for (int i = 0; i < h_f2.size(); ++i) {
    //     std::cout << "F2[" << i << "] = " << h_r[i] << "\n";
    // }

    double h = 1.0 / n2h;
    double hs = h * h;

    blockSize = (n2h * n2h + blockSize - 1) / blockSize;
    
    // Discretizing A to a smaller grid using constants to model the original Grid
    
    discretizeGrid<<<numBlocks, blockSize>>>(d_A2, n2h * n2h, h, hs);
    cudaDeviceSynchronize();

    std::vector<double> h_A2(n2h * n2h, 0.0);
    cudaMemcpy(h_A2.data(), d_A2, n2h * n2h * sizeof(double), cudaMemcpyDeviceToHost);



    if (n2h > X) {
        double *xNew;
        cudaMalloc(&xNew, n2h * sizeof(double));
        std::vector<double> h_x2 = solveJacobi(d_A2, d_x2, d_f2, xNew, n2h);

        cudaMemcpy(d_x2, h_x2.data(), n2h * sizeof(double), cudaMemcpyHostToDevice);

        printf("X vals in N > X\n");
        for (int i = 0; i < h_x2.size(); ++i) {
            printf("Host X[%d] = (%f)\n", i, h_x2[i]);
        }

        double *d_r;
        cudaMalloc(&d_r, n2h * sizeof(double));
        cudaMemset(&d_r, 0.0, n2h * sizeof(double));
        calculateResidual<<<numBlocks, blockSize>>>(d_A2, d_x2, d_f2, d_r, n2h);
        cudaDeviceSynchronize();

        std::vector<double> h_r2(n2h, 0.0);
        cudaMemcpy(h_r2.data(), d_r, n2h * sizeof(double), cudaMemcpyDeviceToHost);

        // printf("Residual in N > X");
        // for (int i = 0; i < h_r2.size(); ++i) {
        //     printf("Host Res[%d] = (%f)\n", i, h_r2[i]);
        // }


        std::vector<double> h_e2 = coarseGridCorrection(A, d_r, tol, n2h);

        

        double *d_e2;
        cudaMalloc(&d_e2, n2h * sizeof(double));
        cudaMemcpy(d_e2, h_e2.data(), n2h * sizeof(double), cudaMemcpyHostToDevice);

        errorCorrection<<<numBlocks, blockSize>>>(d_x2, d_e2, n2h);
        cudaDeviceSynchronize();

        h_x2 = solveJacobi(d_A2, d_f2, d_x2, xNew, n2h);
        cudaMemset(&d_r, 0.0, n2h * sizeof(double));
        calculateResidual<<<numBlocks, blockSize>>>(A, d_x2, d_f2, d_r, n2h);
        cudaDeviceSynchronize();


    } else {
        double *xNew;
        cudaMalloc(&xNew, n2h * sizeof(double));
        std::vector<double> h_x2 = solveJacobi(d_A2, d_f2, d_x2, xNew, n2h);
        //printf("In X < 5\n");
        // for (int i = 0; i < h_x2.size(); ++i) {
        //     printf("X2[%d] = (%f)\n", i, h_x2[i]);
        // }
        cudaMemcpy(d_x2, h_x2.data(), n2h * sizeof(double), cudaMemcpyHostToDevice);

        double *d_r;
        cudaMalloc(&d_r, n2h * sizeof(double));
        cudaMemset(&d_r, 0.0, n2h * sizeof(double));
        calculateResidual<<<numBlocks, blockSize>>>(A, d_x2, d_f2, d_r, n2h);
        cudaDeviceSynchronize();
    }

    double *d_e;
    cudaMalloc(&d_e, N * sizeof(double));

    errorProlongation<<<1,1>>>(d_e, d_x2, N);

    std::vector<double> h_e(N, 0.0);
    
    cudaMemcpy(h_e.data(), d_e, N * sizeof(double), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < h_e.size(); ++i) {
    //         printf("Error[%d] = (%f)\n", i, h_e[i]);
    // }

    cudaFree(d_e);
    cudaFree(d_x2);
    cudaFree(d_A2);
    cudaFree(d_e2);
    cudaFree(d_f2);
    //cudaFree(d_r);


    // Return our error for every level
    return h_e;

}

__global__ void editArray(double *x, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] += 4;
    }
}

void editArrayC(double *x, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] = 4;
    }
}

std::vector<double> multigrid(double* A, double* x, double* b, int N, int levels) {
    double *y = (double*)malloc(5 * sizeof(double));
    memset(y, 0.0, sizeof(y));

    editArrayC(y, 5);
    for (int i = 0; i < 5; ++i) {
        std::cout << y[i] << ",";
    }

    std::cout << "\n";


    double tol = 1e-6;
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    double *xNew;
    cudaMalloc(&xNew, N * sizeof(double));
    
    // Create driver function to solve the Jacobi in the Host 
    std::vector<double> h_x = solveJacobi(A, x, b, xNew, N);

    // Going to need our updated x values from Weighted Jacobi Iteration
    double *d_x;
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    double *d_r;
    cudaMalloc(&d_r, N * sizeof(double));

    calculateResidual<<<numBlocks, blockSize>>>(A, d_x, b, d_r, N);

    std::vector<double> h_r(N, 0.0);
    cudaMemcpy(h_r.data(), d_r, N * sizeof(double), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < h_r.size(); ++i) {
    //     printf("Residual[%d] = (%f)\n", i, h_r[i]);
    // }

    std::vector<double> eh(N, 0.0);
    for (int i = 0; i < 1; ++i) {
        eh = coarseGridCorrection(A, d_r, tol, N);
    }
    


    return eh;
}



__global__ void initVectors(double* x, double* b, int N, double fillValue) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        x[idx] = 0.0;
        b[idx] = fillValue;
    }
}

//#include <chrono>

int main() {
    const int N = 16;
    double *A, *x, *b;
    cudaMalloc(&A, N * N * sizeof(double));
    cudaMalloc(&x, N * sizeof(double));
    cudaMalloc(&b, N * sizeof(double));

    double fillValue = 1.0;
    int blockSize = 256;
    int numBlocks = (N * N + blockSize - 1) / blockSize;
    initMatrixA<<<numBlocks, blockSize>>>(A, N);

    

    numBlocks = (N + blockSize - 1) / blockSize;
    initVectors<<<numBlocks, blockSize>>>(x, b, N, fillValue);

    //std::chrono::high_resolution_clock::time_point beforeMulti = chrono::high_resolution_clock::now();
    std::vector<double> eh = multigrid(A, x, b, N, 4);

    //std::chrono::high_resolution_clock::time_point AfterMulti = chrono::high_resolution_clock::now();

    //float us = (float) std::chrono::duration_cast<std::chrono::microseconds>(beforeMulti-AfterMulti).count();
	//float ms = us/1000;
	//cout << "Time for Kernels (k-means) :" <<ms <<" ms"<<endl;
    
    for (int i = 0; i < eh.size(); ++i) {
        printf("Error[%d] = (%f)\n", i, eh[i]);
    }

    cudaFree(A);
    cudaFree(x);
    cudaFree(b);
    return 0;
}


