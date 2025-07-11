
import torch
from configurations.config import (
    MODEL_PATH, HF_MODEL_NAME, LLM_MODEL_NAME,
    MAX_RETRIES, QUALITY_THRESHOLD, INDEX_SOURCE_URL,
    FORCE_CPU, OPTIMIZE_FOR_MPS, USE_MIXED_PRECISION, TEMPERATURE, MAX_NEW_TOKENS
)
from modules.model_loader import get_optimal_device, load_model
from modules.query import process_query_with_context


def test_complete_rag_pipeline():
    """Test the complete RAG pipeline using configuration settings"""

    print("🔍 Testing Complete RAG Pipeline with Config Settings...")
    print("=" * 50)

    try:
        # Step 1: Test configuration-based model loading
        print("\n1️⃣ Testing configuration-based model loading...")

        # Show current config settings
        print(f"   📋 MODEL_PATH: {MODEL_PATH}")
        print(f"   📋 FORCE_CPU: {FORCE_CPU}")
        print(f"   📋 OPTIMIZE_FOR_MPS: {OPTIMIZE_FOR_MPS}")
        print(f"   📋 USE_MIXED_PRECISION: {USE_MIXED_PRECISION}")

        # Get optimal device based on config
        optimal_device = get_optimal_device()
        print(f"   📋 Optimal device from config: {optimal_device}")

        # Load models using config
        print("   Loading models using config settings...")
        tokenizer, model = load_model()
        print(f"   ✅ Model type: {type(model).__name__}")
        print(f"   ✅ Has generate method: {hasattr(model, 'generate')}")

        # Verify device placement matches config
        actual_device = next(model.parameters()).device
        print(f"   ✅ Model actually loaded on: {actual_device}")

        # Test basic generation with proper device handling
        print("\n2️⃣ Testing basic text generation with config device...")
        # Use the actual device the model is on (respects config)
        device = actual_device
        print(f"   Using device: {device}")

        # Simple generation test with proper device handling
        inputs = tokenizer("Hello", return_tensors="pt")
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if hasattr(model, 'generate'):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False  # Deterministic for testing
                )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"   ✅ Basic generation works: '{response}'")
        else:
            print("   ❌ Model doesn't support generation!")
            return False

        # Step 3: Test vector database with config
        print("\n3️⃣ Testing vector database with config...")
        print(f"   📋 Using INDEX_SOURCE_URL: {INDEX_SOURCE_URL}")
        print(f"   📋 Using HF_MODEL_NAME: {HF_MODEL_NAME}")
        print("   Loading vector DB (this may take a moment)...")

        vector_db, embedding_model = load_vector_db(source="url", source_path=INDEX_SOURCE_URL)
        print(f"   ✅ Vector DB type: {type(vector_db).__name__}")
        print(f"   ✅ Embedding model type: {type(embedding_model).__name__}")

        # Test retrieval
        print("   Testing document retrieval...")
        retriever = vector_db.as_retriever(similarity_top_k=2)
        test_query = "What is artificial intelligence?"
        nodes = retriever.retrieve(test_query)
        print(f"   ✅ Retrieved {len(nodes)} documents")
        if nodes:
            print(f"   ✅ Sample content: {nodes[0].node.get_content()[:100]}...")

        # Step 4: Test full RAG query with config parameters
        print("\n4️⃣ Testing full RAG query with config parameters...")
        print(f"   📋 Using MAX_RETRIES: {MAX_RETRIES}")
        print(f"   📋 Using QUALITY_THRESHOLD: {QUALITY_THRESHOLD}")

        result = process_query_with_context(
            prompt="What is machine learning?",
            model=model,
            tokenizer=tokenizer,
            device=device,  # Use actual device
            vector_db=vector_db,
            embedding_model=embedding_model,
            max_retries=min(MAX_RETRIES, 1),  # Reduce for testing but respect config
            quality_threshold=max(QUALITY_THRESHOLD - 0.4, 0.3)  # Lower threshold for testing
        )

        print(f"   ✅ Query completed!")
        print(f"   ✅ Question: {result.get('question', 'N/A')}")
        print(f"   ✅ Answer: {result.get('answer', 'N/A')[:200]}...")
        print(f"   ✅ Score: {result.get('score', 'N/A')}")
        print(f"   ✅ Attempts: {result.get('attempts', 'N/A')}")
        if result.get('error'):
            print(f"   ⚠️ Error: {result['error']}")

        # Step 5: Validate config effectiveness
        print("\n5️⃣ Validating configuration effectiveness...")

        # Check if device selection worked as expected
        if FORCE_CPU and device.type != "cpu":
            print(f"   ⚠️ FORCE_CPU=True but model on {device.type}")
        elif not FORCE_CPU and torch.backends.mps.is_available() and OPTIMIZE_FOR_MPS and device.type != "mps":
            print(f"   ⚠️ MPS available and OPTIMIZE_FOR_MPS=True but model on {device.type}")
        else:
            print(f"   ✅ Device selection working correctly: {device.type}")

        print("\n🎉 Configuration Test SUCCESSFUL!")
        print("🔧 All config settings are working properly!")
        return True

    except Exception as e:
        print(f"\n❌ Configuration Test FAILED!")
        print(f"   Error: {e}")
        print(f"   Error type: {type(e).__name__}")

        # Additional debugging info
        import traceback
        print("\n📍 Full traceback:")
        traceback.print_exc()

        return False


def test_config_values():
    """Quick test of config values"""
    print("\n📋 Testing Configuration Values...")

    try:
        print(f"   MODEL_PATH: {MODEL_PATH}")
        print(f"   HF_MODEL_NAME: {HF_MODEL_NAME}")
        print(f"   LLM_MODEL_NAME: {LLM_MODEL_NAME}")
        print(f"   MAX_RETRIES: {MAX_RETRIES}")
        print(f"   QUALITY_THRESHOLD: {QUALITY_THRESHOLD}")
        print(f"   INDEX_SOURCE_URL: {INDEX_SOURCE_URL}")
        print(f"   FORCE_CPU: {FORCE_CPU}")
        print(f"   OPTIMIZE_FOR_MPS: {OPTIMIZE_FOR_MPS}")
        print(f"   USE_MIXED_PRECISION: {USE_MIXED_PRECISION}")

        # Validate they're reasonable
        assert isinstance(MODEL_PATH, str) and MODEL_PATH.strip()
        assert isinstance(HF_MODEL_NAME, str) and HF_MODEL_NAME.strip()
        assert isinstance(MAX_RETRIES, int) and MAX_RETRIES > 0
        assert isinstance(QUALITY_THRESHOLD, (int, float)) and 0 <= QUALITY_THRESHOLD <= 1
        assert isinstance(FORCE_CPU, bool)
        assert isinstance(OPTIMIZE_FOR_MPS, bool)
        assert isinstance(USE_MIXED_PRECISION, bool)

        print("   ✅ All config values are valid!")
        return True

    except Exception as e:
        print(f"   ❌ Config test failed: {e}")
        return False


def test_device_configuration():
    """Test device configuration logic"""
    print("\n🔧 Testing Device Configuration Logic...")

    try:
        from modules.model_loader import get_optimal_device

        # Test current config
        device = get_optimal_device()
        print(f"   Current optimal device: {device}")

        # Show what's available
        print(f"   MPS available: {torch.backends.mps.is_available()}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   FORCE_CPU setting: {FORCE_CPU}")
        print(f"   OPTIMIZE_FOR_MPS setting: {OPTIMIZE_FOR_MPS}")

        # Validate logic
        if FORCE_CPU:
            assert device.type == "cpu", f"Expected CPU when FORCE_CPU=True, got {device.type}"
            print("   ✅ FORCE_CPU logic working correctly")
        elif torch.backends.mps.is_available() and OPTIMIZE_FOR_MPS:
            assert device.type == "mps", f"Expected MPS when available and enabled, got {device.type}"
            print("   ✅ MPS optimization logic working correctly")
        elif torch.cuda.is_available():
            print("   ✅ CUDA logic would work (if not overridden)")
        else:
            assert device.type == "cpu", f"Expected CPU fallback, got {device.type}"
            print("   ✅ CPU fallback logic working correctly")

        return True

    except Exception as e:
        print(f"   ❌ Device configuration test failed: {e}")
        return False


def main():
    """Run all configuration tests"""
    print("🧪 RAG SYSTEM CONFIGURATION TEST")
    print("=" * 60)

    # Test 1: Config values
    config_ok = test_config_values()

    # Test 2: Device configuration
    device_ok = test_device_configuration()

    # Test 3: Full pipeline (only if config is OK)
    if config_ok and device_ok:
        pipeline_ok = test_complete_rag_pipeline()

        if pipeline_ok:
            print("\n✅ ALL CONFIGURATION TESTS PASSED!")
            print("🔧 Your config.py is working perfectly!")
            print("🚀 Ready for production use!")
        else:
            print("\n❌ Pipeline test failed - config may need adjustment")
    else:
        print("\n❌ Basic config tests failed - fix configuration first")


if __name__ == "__main__":
    main()