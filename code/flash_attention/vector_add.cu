// Vector addition: OUT[i] = A[i] + B[i]
//
// One thread per element. Each thread reads two floats from HBM and writes
// one. The kernel is memory-bound: peak throughput is the HBM bandwidth.
//
// Build:  nvcc -O3 -arch=native vector_add.cu -o vector_add
// Run  :  ./vector_add

#include "common.cuh"

__global__ void vector_add_kernel(float* OUT, const float* A, const float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) OUT[i] = A[i] + B[i];
}

void vector_add_cpu(float* out, const float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

static void launch(float* d_out, const float* d_a, const float* d_b, int N) {
    constexpr int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;
    vector_add_kernel<<<grid, BLOCK>>>(d_out, d_a, d_b, N);
}

static bool check_correctness() {
    constexpr int N = 1 << 16;
    std::vector<float> hA(N), hB(N), hO(N), hRef(N);
    fill_random(hA.data(), N, 1);
    fill_random(hB.data(), N, 2);
    vector_add_cpu(hRef.data(), hA.data(), hB.data(), N);

    float *dA, *dB, *dO;
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    launch(dO, dA, dB, N);
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, N * sizeof(float), cudaMemcpyDeviceToHost));

    float err = max_abs_diff(hO.data(), hRef.data(), N);
    cudaFree(dA); cudaFree(dB); cudaFree(dO);
    printf("[correctness] N=%d  max abs err=%.2e  %s\n",
           N, err, err < 1e-5f ? "OK" : "FAIL");
    return err < 1e-5f;
}

static void benchmark() {
    int sizes[] = {1 << 16, 1 << 20, 1 << 24, 1 << 26};
    printf("\n%-10s %7s %5s %12s %8s %10s %10s\n",
           "N", "iters", "burs", "med ms/iter", "spread", "GFLOP/s", "GB/s");
    for (int N : sizes) {
        float *dA, *dB, *dO;
        CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dO, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(dA, 0, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(dB, 0, N * sizeof(float)));

        auto s = bench([&] { launch(dO, dA, dB, N); });
        double gflops = (double)N / (s.median_ms * 1e6);
        double gbs    = (double)N * sizeof(float) * 3 / (s.median_ms * 1e6);

        printf("%-10d %7d %5d %12.5f %7.2f%% %10.2f %10.2f\n",
               N, s.iters_per_burst, s.bursts, s.median_ms,
               100.0 * s.spread(), gflops, gbs);
        cudaFree(dA); cudaFree(dB); cudaFree(dO);
    }
}

int main() {
    if (!check_correctness()) return 1;
    benchmark();
    return 0;
}
