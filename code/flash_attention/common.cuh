// Shared utilities for the CUDA building-block demos.
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <algorithm>
#include <cmath>
#include <vector>
#include <chrono>
#include <thread>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err__));            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer()  { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { cudaEventRecord(start_); }
    float stop_ms() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

inline void fill_random(float* p, size_t n, unsigned seed = 0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (size_t i = 0; i < n; ++i) p[i] = dist(rng);
}

inline float max_abs_diff(const float* a, const float* b, size_t n) {
    float m = 0.f;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// Microbenchmark a kernel.
//
// Strategy:
//   1. Warm up by launching the kernel until `warmup_ms` of wall time has
//      passed -- enough to bring the GPU out of low-power idle and to settle
//      thermals so boost clocks stabilise.
//   2. Time one launch to estimate cost, then pick `iters_per_burst` so each
//      burst takes ~`target_burst_ms` (long enough to dwarf event overhead).
//   3. Run `bursts` independent bursts and return min / median / max of the
//      per-iter times. The min is the cleanest signal (one-sided noise),
//      median is a fair "typical" number, max-min/median is the spread.
struct BenchStats {
    float min_ms;
    float median_ms;
    float max_ms;
    int iters_per_burst;
    int bursts;
    float spread() const { return (max_ms - min_ms) / median_ms; }
};

template <typename F>
BenchStats bench(F&& fn,
                 double target_burst_ms = 50.0,
                 int bursts = 5,
                 double warmup_ms = 100.0) {
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
        fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto el = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (el >= warmup_ms) break;
    }

    GpuTimer ct; ct.start(); fn(); float one_ms = ct.stop_ms();
    int iters = std::max(1, (int)std::ceil(target_burst_ms / std::max(1e-3f, one_ms)));

    std::vector<float> samples;
    samples.reserve(bursts);
    for (int b = 0; b < bursts; ++b) {
        GpuTimer t; t.start();
        for (int i = 0; i < iters; ++i) fn();
        samples.push_back(t.stop_ms() / iters);
    }
    std::sort(samples.begin(), samples.end());
    return { samples.front(), samples[bursts / 2], samples.back(), iters, bursts };
}
