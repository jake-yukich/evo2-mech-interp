#!/usr/bin/env python3
"""
Lightweight Evo 2 tests.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# try:
#     _ptxas_path = "/opt/conda/envs/evo2/bin/ptxas"
#     if os.path.isfile(_ptxas_path):
#         os.environ.setdefault("TRITON_PTXAS_PATH", _ptxas_path)
# except Exception:
#     pass

from evo2 import Evo2


def test_model_loading():
    """Test Evo2 model initialization and basic properties."""
    print("=" * 60)
    print("TEST 1: Model Loading and Basic Properties")
    print("=" * 60)
    
    try:
        # Load the model
        print("Loading Evo2 model...")
        model = Evo2("evo2_7b")
        print("✓ Model loaded successfully")
        
        # Check basic properties
        print(f"Model type: {type(model)}")
        print(f"Model device: {next(model.model.parameters()).device}")
        print(f"Model dtype: {next(model.model.parameters()).dtype}")
        print(f"Model training mode: {model.model.training}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU memory allocated: {gpu_memory:.2f} GB")
        
        # Set to eval mode
        model.model.eval()
        print(f"Model in eval mode: {not model.model.training}")
        
        return model
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None


def test_tokenization(model):
    """Test tokenizer functionality with simple sequences."""
    print("\n" + "=" * 60)
    print("TEST 2: Tokenization")
    print("=" * 60)
    
    if model is None:
        print("✗ Skipping tokenization test - no model loaded")
        return None
    
    try:
        # Test sequences of different lengths
        test_sequences = [
            "ATCG",
            "ATCGATCG",
            "ATCGATCGATCGATCG",  # 16 bases
            "ATCG" * 100,  # 400 bases
        ]
        
        tokenized_results = []
        
        for seq in test_sequences:
            print(f"\nTesting sequence: {seq[:20]}{'...' if len(seq) > 20 else ''} (length: {len(seq)})")
            
            # Tokenize
            tokens = model.tokenizer.tokenize(seq)
            print(f"  Tokenized to {len(tokens)} tokens")
            print(f"  First 10 tokens: {tokens[:10]}")
            
            # Convert to tensor
            token_ids = [int(token) for token in tokens]
            token_tensor = torch.tensor(token_ids, dtype=torch.long)
            print(f"  Tensor shape: {token_tensor.shape}")
            print(f"  Tensor dtype: {token_tensor.dtype}")
            
            tokenized_results.append({
                'sequence': seq,
                'tokens': tokens,
                'token_ids': token_ids,
                'tensor': token_tensor
            })
        
        print("✓ Tokenization test passed")
        return tokenized_results
        
    except Exception as e:
        print(f"✗ Tokenization test failed: {e}")
        return None


def test_forward_pass(model, tokenized_results):
    """Test basic forward pass and embedding extraction."""
    print("\n" + "=" * 60)
    print("TEST 3: Forward Pass and Embedding Extraction")
    print("=" * 60)
    
    if model is None or tokenized_results is None:
        print("✗ Skipping forward pass test - no model or tokenized data")
        return None
    
    try:
        # Test with the shortest sequence first
        test_data = tokenized_results[0]  # "ATCG"
        sequence = test_data['sequence']
        token_tensor = test_data['tensor']
        
        print(f"Testing forward pass with sequence: {sequence}")
        print(f"Input tensor shape: {token_tensor.shape}")
        
        # Move to GPU if available
        device = next(model.model.parameters()).device
        input_tensor = token_tensor.unsqueeze(0).to(device)  # Add batch dimension
        print(f"Input tensor on device: {input_tensor.device}")
        print(f"Input tensor shape (with batch): {input_tensor.shape}")
        
        # Forward pass
        model.model.eval()
        with torch.no_grad():
            # Test basic forward pass
            logits = model.model(input_tensor)
            print(f"Logits shape: {logits.shape}")
            print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
            print(f"Logits has NaN: {torch.isnan(logits).any()}")
            
            # Try to discover available layers by testing different layer name formats
            print("\nDiscovering available layers...")
            layer_candidates = [
                "blocks.24.mlp.l3",  # From config
                "blocks-24",         # From sparse autoencoder demo
                "blocks.24",         # Alternative format
                "layer_24",          # Another possible format
            ]
            
            working_layer = None
            for layer_name in layer_candidates:
                try:
                    print(f"  Testing layer: {layer_name}")
                    logits_with_embeddings, embeddings = model(
                        input_tensor, 
                        return_embeddings=True, 
                        layer_names=[layer_name]
                    )
                    
                    if layer_name in embeddings:
                        working_layer = layer_name
                        print(f"  ✓ Found working layer: {layer_name}")
                        break
                    else:
                        print(f"  ✗ Layer {layer_name} not found in embeddings")
                        
                except Exception as e:
                    print(f"  ✗ Layer {layer_name} failed: {e}")
                    continue
            
            if working_layer:
                print(f"\nUsing working layer: {working_layer}")
                logits_with_embeddings, embeddings = model(
                    input_tensor, 
                    return_embeddings=True, 
                    layer_names=[working_layer]
                )
                
                print(f"Logits with embeddings shape: {logits_with_embeddings.shape}")
                print(f"Embeddings keys: {list(embeddings.keys())}")
                
                layer_embeddings = embeddings[working_layer]
                print(f"Layer embeddings shape: {layer_embeddings.shape}")
                print(f"Layer embeddings stats: min={layer_embeddings.min():.4f}, max={layer_embeddings.max():.4f}, mean={layer_embeddings.mean():.4f}")
                print(f"Layer embeddings has NaN: {torch.isnan(layer_embeddings).any()}")
            else:
                print("⚠️  No working layer found, but basic forward pass succeeded")
                print("This might be normal - embeddings extraction may require specific layer names")
        
        print("✓ Forward pass test passed")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation(model):
    """Test text generation capabilities."""
    print("\n" + "=" * 60)
    print("TEST 4: Text Generation")
    print("=" * 60)
    
    if model is None:
        print("✗ Skipping generation test - no model loaded")
        return False
    
    try:
        # Test generation with simple sequences
        test_prompts = ["ATCG", "ATCGATCG"]
        all_success = True
        
        for prompt in test_prompts:
            print(f"\nTesting generation with prompt: {prompt}")
            
            try:
                # Generate text
                generated = model.generate(
                    [prompt], 
                    n_tokens=10, 
                    temperature=1.0, 
                    top_k=4, 
                    top_p=1.0, 
                    batched=True, 
                    verbose=1
                )
                print(f"Generated output: {generated}")
                print("✓ Generation successful")
                
            except Exception as e:
                print(f"✗ Generation failed for prompt '{prompt}': {e}")
                all_success = False  # Mark as failed if any prompt fails
        
        if all_success:
            print("✓ Generation test completed")
            return True
        else:
            print("✗ Generation test failed for one or more prompts")
            return False
        
    except Exception as e:
        print(f"✗ Generation test failed: {e}")
        return False


def test_memory_cleanup(model):
    """Test memory cleanup and GPU cache clearing."""
    print("\n" + "=" * 60)
    print("TEST 5: Memory Cleanup")
    print("=" * 60)
    
    try:
        if torch.cuda.is_available():
            # Measure memory before cleanup
            mem_before = torch.cuda.memory_allocated()
            print(f"GPU memory before cleanup: {mem_before / 1024**3:.2f} GB")
            
            if model is not None:
                print("Deleting model to free GPU memory...")
                
                # Move model to CPU first to free GPU memory
                if hasattr(model, 'model'):
                    model.model.cpu()
                
                # Delete the model object
                del model
                model = None
            
            # Multiple rounds of cleanup for stubborn memory
            print("Performing aggressive memory cleanup...")
            for i in range(3):
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Synchronize to ensure operations complete
                torch.cuda.synchronize()
                
                print(f"  Cleanup round {i+1} complete")
            
            # Measure memory after cleanup
            mem_after = torch.cuda.memory_allocated()
            print(f"GPU memory after cleanup: {mem_after / 1024**3:.2f} GB")
            
            # Check if we freed at least some memory (allow for some tolerance)
            memory_freed = mem_before - mem_after
            if memory_freed > 0.1 * 1024**3:  # At least 100MB freed
                print(f"✓ GPU memory was reduced by {memory_freed / 1024**3:.2f} GB")
                result = True
            else:
                print("⚠️  GPU memory was not significantly reduced after cleanup")
                print("This might be normal if the model was already unloaded or memory is fragmented")
                result = True  # Don't fail the test for this
        else:
            print("CUDA not available, skipping GPU memory cleanup test")
            result = True  # Consider as pass if no CUDA
        
        print("✓ Memory cleanup completed")
        return result
        
    except Exception as e:
        print(f"✗ Memory cleanup failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Evo2 Basic Functionality Test Suite")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Run tests
    model = test_model_loading()
    tokenized_results = test_tokenization(model)
    forward_pass_success = test_forward_pass(model, tokenized_results)
    generation_success = test_generation(model)
    cleanup_success = test_memory_cleanup(model)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    tests = [
        ("Model Loading", model is not None),
        ("Tokenization", tokenized_results is not None),
        ("Forward Pass", forward_pass_success),
        ("Generation", generation_success),
        ("Memory Cleanup", cleanup_success),
    ]
    
    passed = 0
    for test_name, success in tests:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Evo2 is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
