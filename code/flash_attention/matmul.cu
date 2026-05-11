// Matrix multiplication: C = A * B
//   A is M x K, B is K x N, C is M x N (all row-major).
//
// Two kernels:
//
//   naive : one thread per output element. Each thread streams a row of A
//           and a column of B from HBM. Every element of A is reread N
//           times, every element of B reread M times. The L1/L2 cache
//           recovers most of that reuse on Ada, so it isn't a disaster --
//           but the kernel hits a small fraction of fp32 peak.
//
//   tiled : each block computes a TILE x TILE tile of C. The K dimension
//           is streamed in TILE-wide chunks: each iteration cooperatively
//           loads one TILE x TILE block of A and one of B from HBM into
//           shared memory, then every thread accumulates one partial dot
//           product from those tiles. With TILE=32, each global element
//           of A/B is reused 32 times from SRAM instead of being reread
//           from HBM.
//
// This is the same "load-once, compute-many" pattern that Flash Attention
// applies to Q,K,V tiles for the attention matmul.
//
// Build: nvcc -O3 -arch=native matmul.cu -o matmul
// Run  : ./matmul

#include "common.cuh"

constexpr int TILE = 32;

__global__ void matmul_naive_kernel(float* C, const float* A, const float* B,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

__global__ void matmul_tiled_kernel(float* C, const float* A, const float* B,
                                    int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.f;
    for (int kt = 0; kt < K; kt += TILE) {
        int a_col = kt + threadIdx.x;
        int b_row = kt + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

void matmul_cpu(float* C, const float* A, const float* B, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

static void launch_naive(float* dC, const float* dA, const float* dB,
                         int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_naive_kernel<<<grid, block>>>(dC, dA, dB, M, N, K);
}

static void launch_tiled(float* dC, const float* dA, const float* dB,
                         int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<grid, block>>>(dC, dA, dB, M, N, K);
}

static bool check_correctness() {
    constexpr int M = 200, N = 250, K = 96; // not multiples of TILE
    std::vector<float> hA(M * K), hB(K * N), hN(M * N), hT(M * N), hRef(M * N);
    fill_random(hA.data(), M * K, 1);
    fill_random(hB.data(), K * N, 2);
    matmul_cpu(hRef.data(), hA.data(), hB.data(), M, N, K);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    launch_naive(dC, dA, dB, M, N, K);
    CUDA_CHECK(cudaMemcpy(hN.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    launch_tiled(dC, dA, dB, M, N, K);
    CUDA_CHECK(cudaMemcpy(hT.data(), dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    float en = max_abs_diff(hN.data(), hRef.data(), M * N);
    float et = max_abs_diff(hT.data(), hRef.data(), M * N);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    float tol = 1e-3f;
    printf("[correctness] %dx%d * %dx%d  naive err=%.2e  tiled err=%.2e  %s\n",
           M, K, K, N, en, et, (en < tol && et < tol) ? "OK" : "FAIL");
    return en < tol && et < tol;
}

static void benchmark() {
    int sizes[] = {256, 512, 1024, 2048};
    printf("\n%-8s %12s %8s %10s %12s %8s %10s %10s\n",
           "M=N=K", "naive ms", "spread", "naive GF/s",
           "tiled ms", "spread", "tiled GF/s", "speedup");
    for (int S : sizes) {
        size_t bytes = (size_t)S * S * sizeof(float);
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
        CUDA_CHECK(cudaMemset(dA, 0, bytes));
        CUDA_CHECK(cudaMemset(dB, 0, bytes));

        auto sn = bench([&] { launch_naive(dC, dA, dB, S, S, S); });
        auto st = bench([&] { launch_tiled(dC, dA, dB, S, S, S); });
        double gf_naive = 2.0 * (double)S * S * S / (sn.median_ms * 1e6);
        double gf_tiled = 2.0 * (double)S * S * S / (st.median_ms * 1e6);

        printf("%-8d %12.5f %7.2f%% %10.2f %12.5f %7.2f%% %10.2f %9.2fx\n",
               S, sn.median_ms, 100.0 * sn.spread(), gf_naive,
               st.median_ms, 100.0 * st.spread(), gf_tiled,
               sn.median_ms / st.median_ms);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
}

int main() {
    if (!check_correctness()) return 1;
    benchmark();
    return 0;
}
