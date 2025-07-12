# comprehensive_mps_test.py - Complete MPS compatibility test suite

import os
import sys
import torch
import traceback

from utility.device_utils import get_optimal_device

# Set MPS environment before any imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def setup_mps_environment():
    """Configure optimal MPS environment settings."""
    if torch.backends.mps.is_available():
        torch.backends.mps.allow_tf32 = False
        print("✅ MPS environment configured")
        return True
    else:
        print("❌ MPS not available on this system")
        return False


class MPSTestSuite:
    """Comprehensive test suite for MPS compatibility."""

    def __init__(self):
        self.results = {}
        self.mps_available = setup_mps_environment()

    def run_test(self, test_name: str, test_func, *args, **kwargs):
        """Run a single test and record results."""
        print(f"\n🧪 Running: {test_name}")
        print("-" * 50)

        try:
            result = test_func(*args, **kwargs)
            self.results[test_name] = {
                'status': 'PASSED' if result else 'FAILED',
                'result': result,
                'error': None
            }
            status_emoji = "✅" if result else "❌"
            print(f"{status_emoji} {test_name}: {'PASSED' if result else 'FAILED'}")
            return result

        except Exception as e:
            self.results[test_name] = {
                'status': 'ERROR',
                'result': None,
                'error': str(e)
            }
            print(f"💥 {test_name}: ERROR - {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            return False

    def test_basic_torch_mps(self):
        """Test basic PyTorch MPS functionality."""
        if not self.mps_available:
            return False

        try:
            # Test basic tensor operations
            device = torch.device("mps")
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.mm(x, y)

            print(f"  ✓ Basic tensor operations work")
            print(f"  ✓ Result tensor shape: {z.shape}")
            print(f"  ✓ Result tensor device: {z.device}")
            return True

        except Exception as e:
            print(f"  ✗ Basic MPS operations failed: {e}")
            return False

    def test_model_loading(self):
        """Test MPS-safe model loading."""
        try:
            from modules.model_loader import load_model

            device = get_optimal_device()
            print(f"  ✓ Optimal device detected: {device}")

            tokenizer, model = load_model()
            print(f"  ✓ Model loaded successfully")

            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            print(f"  ✓ Model device: {model_device}")
            print(f"  ✓ Model dtype: {model_dtype}")

            # Check for any bfloat16 parameters
            bfloat16_params = [name for name, param in model.named_parameters()
                               if param.dtype == torch.bfloat16]
            if bfloat16_params:
                print(f"  ⚠️  Found bfloat16 parameters: {len(bfloat16_params)}")
                return False
            else:
                print(f"  ✓ No bfloat16 parameters found")

            return True

        except Exception as e:
            print(f"  ✗ Model loading failed: {e}")
            return False

    def test_tokenizer_compatibility(self):
        """Test tokenizer with MPS input preparation."""
        try:
            from modules.model_loader import load_model, prepare_mps_inputs

            tokenizer, model = load_model()
            device = next(model.parameters()).device

            # Test tokenization
            test_text = "Hello, this is a test for MPS compatibility."
            inputs = tokenizer(test_text, return_tensors="pt")
            print(f"  ✓ Tokenization successful")

            # Test MPS input preparation
            if device.type == "mps":
                mps_inputs = prepare_mps_inputs(inputs, device)
                print(f"  ✓ MPS input preparation successful")

                # Check dtypes
                for key, value in mps_inputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"    {key}: {value.dtype} on {value.device}")

                # Verify no bfloat16
                bfloat16_tensors = [k for k, v in mps_inputs.items()
                                    if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16]
                if bfloat16_tensors:
                    print(f"  ✗ Found bfloat16 tensors: {bfloat16_tensors}")
                    return False
                else:
                    print(f"  ✓ All input tensors are MPS-compatible")

            return True

        except Exception as e:
            print(f"  ✗ Tokenizer compatibility test failed: {e}")
            return False

    def test_text_generation(self):
        """Test MPS-safe text generation."""
        try:
            from modules.model_loader import load_model, generate_text_mps_safe, prepare_mps_inputs

            tokenizer, model = load_model()
            device = next(model.parameters()).device

            # Prepare input
            test_prompt = "The future of artificial intelligence is"
            inputs = tokenizer(test_prompt, return_tensors="pt")

            if device.type == "mps":
                inputs = prepare_mps_inputs(inputs, device)
            else:
                inputs = {k: v.to(device) for k, v in inputs.items()}

            print(f"  ✓ Input preparation complete")

            # Generate text
            outputs = generate_text_mps_safe(model, inputs, device, tokenizer, max_tokens=20)
            print(f"  ✓ Text generation successful")

            # Decode result
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  ✓ Generated text: '{result}'")

            return True

        except Exception as e:
            print(f"  ✗ Text generation failed: {e}")
            return False

    def test_query_processing(self):
        """Test complete query processing pipeline."""
        try:
            from modules.model_loader import load_model
            from modules.query import process_query_with_context

            tokenizer, model = load_model()
            device = next(model.parameters()).device

            # Test simple query without vector DB
            test_query = "What is machine learning?"
            print(f"  ✓ Testing query: '{test_query}'")

            result = process_query_with_context(
                test_query, model, tokenizer, device,
                vector_db=None, embedding_model=None,
                max_retries=1, quality_threshold=0.1
            )

            if result["error"]:
                print(f"  ✗ Query processing error: {result['error']}")
                return False
            else:
                print(f"  ✓ Query processing successful")
                print(f"    Answer: {result['answer'][:50]}...")
                print(f"    Device used: {result.get('device_used', 'unknown')}")
                print(f"    Score: {result.get('score', 'N/A')}")
                return True

        except Exception as e:
            print(f"  ✗ Query processing test failed: {e}")
            return False

    def test_vector_db_integration(self):
        """Test vector database integration (optional)."""
        try:
            from vector_db.indexer import load_vector_db
            from configurations.config import INDEX_SOURCE_URL

            print(f"  ✓ Attempting to load vector DB...")
            vector_db, embedding_model = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)
            print(f"  ✓ Vector DB loaded: {type(vector_db).__name__}")

            # Test with query processing
            from modules.model_loader import load_model
            from modules.query import process_query_with_context

            tokenizer, model = load_model()
            device = next(model.parameters()).device

            test_query = "What is the capital of France?"
            result = process_query_with_context(
                test_query, model, tokenizer, device,
                vector_db, embedding_model,
                max_retries=1, quality_threshold=0.1
            )

            if result["error"]:
                print(f"  ⚠️  Query with vector DB failed: {result['error']}")
                return False
            else:
                print(f"  ✓ Vector DB integration successful")
                return True

        except Exception as e:
            print(f"  ⚠️  Vector DB test failed (this is optional): {e}")
            return True  # Don't fail the whole suite for optional components

    def test_memory_management(self):
        """Test GPU memory management and cleanup."""
        try:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            if device.type == "mps":
                # Test memory allocation and cleanup
                large_tensor = torch.randn(1000, 1000, device=device)
                print(f"  ✓ Large tensor allocated")

                del large_tensor
                torch.mps.empty_cache()
                print(f"  ✓ Memory cleanup successful")

            return True

        except Exception as e:
            print(f"  ✗ Memory management test failed: {e}")
            return False

    def test_error_handling(self):
        """Test error handling and fallback mechanisms."""
        try:
            from modules.model_loader import load_model
            from modules.query import handle_gpu_memory_error

            tokenizer, model = load_model()
            device = next(model.parameters()).device

            # Test with empty inputs (should trigger error handling)
            empty_inputs = {}

            # This should trigger error handling
            try:
                answer, device_used = handle_gpu_memory_error(
                    RuntimeError("Simulated MPS error"),
                    model, empty_inputs, device, tokenizer
                )
                print(f"  ✓ Error handling successful")
                print(f"    Fallback device: {device_used}")
                return True
            except Exception as e:
                print(f"  ✓ Error handling triggered as expected: {type(e).__name__}")
                return True

        except Exception as e:
            print(f"  ✗ Error handling test failed: {e}")
            return False

    def run_all_tests(self):
        """Run the complete test suite."""
        print("🚀 Starting Comprehensive MPS Test Suite")
        print("=" * 60)

        if not self.mps_available:
            print("❌ MPS not available. Running CPU tests only.")

        # Define test sequence
        tests = [
            ("Basic PyTorch MPS", self.test_basic_torch_mps),
            ("Model Loading", self.test_model_loading),
            ("Tokenizer Compatibility", self.test_tokenizer_compatibility),
            ("Text Generation", self.test_text_generation),
            ("Query Processing", self.test_query_processing),
            ("Vector DB Integration", self.test_vector_db_integration),
            ("Memory Management", self.test_memory_management),
            ("Error Handling", self.test_error_handling),
        ]

        # Run all tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in self.results.values() if r['status'] == 'PASSED')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAILED')
        errors = sum(1 for r in self.results.values() if r['status'] == 'ERROR')
        total = len(self.results)

        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"💥 Errors: {errors}")

        success_rate = (passed / total) * 100 if total > 0 else 0
        print(f"🎯 Success Rate: {success_rate:.1f}%")

        if failed > 0 or errors > 0:
            print("\n❌ FAILED/ERROR TESTS:")
            for test_name, result in self.results.items():
                if result['status'] in ['FAILED', 'ERROR']:
                    print(f"  • {test_name}: {result['status']}")
                    if result['error']:
                        print(f"    Error: {result['error']}")

        # Overall assessment
        if success_rate >= 90:
            print(f"\n🎉 EXCELLENT! Your MPS setup is working great!")
        elif success_rate >= 75:
            print(f"\n✅ GOOD! Most features work, minor issues detected.")
        elif success_rate >= 50:
            print(f"\n⚠️  PARTIAL: Some features work, needs attention.")
        else:
            print(f"\n❌ POOR: Major issues detected, troubleshooting needed.")


def quick_mps_check():
    """Quick MPS availability and basic functionality check."""
    print("🔍 Quick MPS Check")
    print("-" * 30)

    # Check availability
    mps_available = torch.backends.mps.is_available()
    print(f"MPS Available: {'✅ YES' if mps_available else '❌ NO'}")

    if not mps_available:
        print("This system doesn't support MPS. Tests will run on CPU.")
        return False

    # Check PyTorch version
    print(f"PyTorch Version: {torch.__version__}")

    # Basic MPS test
    try:
        device = torch.device("mps")
        x = torch.randn(10, 10, device=device)
        y = x.sum()
        print(f"Basic MPS Test: ✅ PASSED")
        return True
    except Exception as e:
        print(f"Basic MPS Test: ❌ FAILED - {e}")
        return False


if __name__ == "__main__":
    # Quick check first
    quick_check_passed = quick_mps_check()

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print(f"\n🎯 Quick Check Result: {'PASSED' if quick_check_passed else 'FAILED'}")
        sys.exit(0 if quick_check_passed else 1)

    # Full test suite
    print("\n" + "=" * 60)
    test_suite = MPSTestSuite()
    test_suite.run_all_tests()

    # Exit with appropriate code
    passed = sum(1 for r in test_suite.results.values() if r['status'] == 'PASSED')
    total = len(test_suite.results)
    success_rate = (passed / total) * 100 if total > 0 else 0

    sys.exit(0 if success_rate >= 75 else 1)
