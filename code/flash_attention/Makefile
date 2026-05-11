NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -arch=native -std=c++17

BINS = vector_add matrix_add matmul

.PHONY: all clean run
all: $(BINS)

%: %.cu common.cuh
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: all
	@for b in $(BINS); do echo "===== $$b ====="; ./$$b; echo; done

clean:
	rm -f $(BINS)
