# Makefile for CUDA memory migration benchmarks
# Compiles first_touch.cu and hot-pages.cu programs

# Compiler and flags
NVCC = nvcc
CFLAGS = -O3 -arch=sm_90 -I/usr/include/numa
LDFLAGS = -lnuma -lcuda

# Source files (relative to current directory)
FIRST_TOUCH_SRC = first_touch.cu
HOT_PAGES_SRC = eviction.cu

# Output executables
FIRST_TOUCH_BIN = first_touch
HOT_PAGES_BIN = eviction

# Default target - build both programs
.PHONY: all clean first_touch hot_pages run_first_touch run_hot_pages

all: $(FIRST_TOUCH_BIN) $(HOT_PAGES_BIN)

# Build first_touch program
$(FIRST_TOUCH_BIN): $(FIRST_TOUCH_SRC)
	@echo "Cleaning old first_touch binary..."
	@rm -f $(FIRST_TOUCH_BIN)
	@echo "Compiling first_touch.cu..."
	$(NVCC) $(CFLAGS) -o $(FIRST_TOUCH_BIN) $(FIRST_TOUCH_SRC) $(LDFLAGS)
	@echo "✓ first_touch compiled successfully"

# Build hot_pages program  
$(HOT_PAGES_BIN): $(HOT_PAGES_SRC)
	@echo "Cleaning old hot_pages binary..."
	@rm -f $(HOT_PAGES_BIN)
	@echo "Compiling hot-pages.cu..."
	$(NVCC) $(CFLAGS) -o $(HOT_PAGES_BIN) $(HOT_PAGES_SRC) $(LDFLAGS)
	@echo "✓ hot_pages compiled successfully"

# Individual targets
first_touch: $(FIRST_TOUCH_BIN)

hot_pages: $(HOT_PAGES_BIN)

# Run targets
run_first_touch: $(FIRST_TOUCH_BIN)
	@echo "Running first_touch benchmark..."
	./$(FIRST_TOUCH_BIN)

run_hot_pages: $(HOT_PAGES_BIN)
	@echo "Running hot_pages benchmark..."
	./$(HOT_PAGES_BIN)

# Clean all binaries
clean:
	@echo "Cleaning all binaries..."
	@rm -f $(FIRST_TOUCH_BIN) $(HOT_PAGES_BIN)
	@echo "✓ Clean complete"

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build both programs (default)"
	@echo "  first_touch      - Build only first_touch program"
	@echo "  eviction        - Build only hot_pages program"
	@echo "  run_first_touch  - Build and run first_touch"
	@echo "  run_hot_pages    - Build and run hot_pages"
	@echo "  clean            - Remove all compiled binaries"
	@echo "  help             - Show this help message" 