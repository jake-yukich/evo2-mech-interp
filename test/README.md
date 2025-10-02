# Evo2 Test Suite

This directory contains test scripts for verifying Evo2 functionality.

## test_evo2_basic.py

A lightweight test script that verifies the core Evo2 capabilities without relying on wrapper functions. This script tests:

1. **Model Loading**: Initializes the Evo2 model and checks basic properties
2. **Tokenization**: Tests the tokenizer with sequences of different lengths
3. **Forward Pass**: Tests basic inference and embedding extraction
4. **Text Generation**: Tests the model's generation capabilities
5. **Memory Cleanup**: Verifies proper GPU memory management

### Usage

```bash
# Run the test script
python test_evo2_basic.py

# Or make it executable and run directly
chmod +x test_evo2_basic.py
./test_evo2_basic.py
```

### Requirements

- CUDA-capable GPU (recommended)
- Evo2 model installed and accessible
- Sufficient GPU memory for the 7B model

### Expected Output

The script will run through all tests and provide a summary at the end. All tests should pass for Evo2 to be considered working correctly.
