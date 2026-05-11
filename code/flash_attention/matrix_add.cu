// Matrix addition: OUT[r,c] = A[r,c] + B[r,c]
//
// Two kernels with the same arithmetic intensity but opposite indexing:
//
//   coalesced: threadIdx.x -> col   (warp's fast axis matches the contiguous
//              memory axis -> 32 floats per warp = one 128-byte transaction).
//   strided  : threadIdx.x -> row   (warp's fast axis steps by `cols` floats
//              -> 32 separate transactions per warp).
//
// Same data layout (row-major), same FLOPs and bytes touched. Only the access
// pattern differs. The slowdown isolates the cost of uncoalesced HBM access.
//
// Build: nvcc -O3 -arch=native matrix_add.cu -o matrix_add
// Run  : ./matrix_add

#include "common.cuh"

// Coalesced: consecutive threads in a warp hit consecutive memory addresses.
__global__ void matrix_add_coalesced_kernel(float* OUT, const float* A,
                                            const float* B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        size_t idx = (size_t)row * cols + col;
        OUT[idx] = A[idx] + B[idx];
    }
}

// Strided: swap the role of threadIdx.x and threadIdx.y. Within a warp,
// consecutive threads now differ in `row`, so their addresses are
// `cols` floats apart -- non-coalesced.
__global__ void matrix_add_strided_kernel(float* OUT, const float* A,
                                          const float* B, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols) {
        size_t idx = (size_t)row * cols + col;
        OUT[idx] = A[idx] + B[idx];
    }
}

void matrix_add_cpu(float* out, const float* a, const float* b, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) out[i] = a[i] + b[i];
}

static void launch_coalesced(float* d_out, const float* d_a, const float* d_b,
                             int rows, int cols, dim3 block) {
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);
    matrix_add_coalesced_kernel<<<grid, block>>>(d_out, d_a, d_b, rows, cols);
}

static void launch_strided(float* d_out, const float* d_a, const float* d_b,
                           int rows, int cols, dim3 block) {
    dim3 grid((rows + block.x - 1) / block.x,
              (cols + block.y - 1) / block.y);
    matrix_add_strided_kernel<<<grid, block>>>(d_out, d_a, d_b, rows, cols);
}

static bool check_correctness() {
    constexpr int R = 257, C = 513; // deliberately not multiples of 16
    int N = R * C;
    std::vector<float> hA(N), hB(N), hC(N), hS(N), hRef(N);
    fill_random(hA.data(), N, 1);
    fill_random(hB.data(), N, 2);
    matrix_add_cpu(hRef.data(), hA.data(), hB.data(), R, C);

    float *dA, *dB, *dO;
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    launch_coalesced(dO, dA, dB, R, C, dim3(16, 16));
    CUDA_CHECK(cudaMemcpy(hC.data(), dO, N * sizeof(float), cudaMemcpyDeviceToHost));
    launch_strided(dO, dA, dB, R, C, dim3(16, 16));
    CUDA_CHECK(cudaMemcpy(hS.data(), dO, N * sizeof(float), cudaMemcpyDeviceToHost));

    float ec = max_abs_diff(hC.data(), hRef.data(), N);
    float es = max_abs_diff(hS.data(), hRef.data(), N);
    cudaFree(dA); cudaFree(dB); cudaFree(dO);
    printf("[correctness] %dx%d  coalesced err=%.2e  strided err=%.2e  %s\n",
           R, C, ec, es, (ec < 1e-5f && es < 1e-5f) ? "OK" : "FAIL");
    return ec < 1e-5f && es < 1e-5f;
}

// Block-shape sweep at a single matrix size, in the deep HBM-bound regime.
//
// What matters for HBM bandwidth is the *warp's* access pattern, which is
// determined by the combination of kernel (which thread index drives row vs
// col) AND block shape (which thread axes are inside one warp). A "warp"
// is 32 threads laid out in linearization order (tx fastest, then ty), so
// block.x effectively decides how many lanes of a warp share row/col.
static void benchmark() {
    constexpr int R = 4096, C = 4096; // 192 MB working set, far above L2
    size_t N = (size_t)R * C;
    size_t bytes = N * sizeof(float) * 3;
    float *dA, *dB, *dO;
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(dA, 0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(dB, 0, N * sizeof(float)));

    struct Cfg { int bx, by; const char* note; };
    Cfg configs[] = {
        {  32,   32, "1024 thr / 32 warps"},
        {  32,    8, " 256 thr /  8 warps"},
        {  64,    1, "  64 thr /  2 warps"},
        {   1,   64, "  64 thr /  2 warps"},
        {  16,   16, " 256 thr /  8 warps"},
        {   8,   32, " 256 thr /  8 warps"},
        { 256,    1, " 256 thr /  8 warps"},
        {   1,  256, " 256 thr /  8 warps"},
        { 128,    1, " 128 thr /  4 warps"},
        {   1,  128, " 128 thr /  4 warps"},
        {  32,    1, "  32 thr /  1 warp "},
        {   1,   32, "  32 thr /  1 warp "},
        {   1,    1, "   1 thr / 1/32 wp"},
    };

    // Cool the GPU between configs so the 4070's boost clock doesn't
    // drift mid-sweep. Without this, later rows read low because the
    // chip is throttling, not because of the access pattern.
    auto cool = [] {
        CUDA_CHECK(cudaDeviceSynchronize());
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    };

    printf("\nBlock-shape sweep at %dx%d  (HBM peak ~672 GB/s)\n", R, C);
    printf("%-8s %-22s %14s %14s\n", "block", "", "coal GB/s", "strd GB/s");
    for (auto cfg : configs) {
        dim3 block(cfg.bx, cfg.by);
        cool(); auto sc = bench([&] { launch_coalesced(dO, dA, dB, R, C, block); });
        cool(); auto ss = bench([&] { launch_strided  (dO, dA, dB, R, C, block); });
        char buf[32]; snprintf(buf, sizeof(buf), "%dx%d", cfg.bx, cfg.by);
        printf("%-8s %-22s %14.1f %14.1f\n", buf, cfg.note,
               (double)bytes / (sc.median_ms * 1e6),
               (double)bytes / (ss.median_ms * 1e6));
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dO);
}

int main() {
    if (!check_correctness()) return 1;
    benchmark();
    return 0;
}
