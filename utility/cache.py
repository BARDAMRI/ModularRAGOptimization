# utility/cache.py
import json
import pickle
import hashlib
import time
import os
from pathlib import Path
from typing import Any, Optional, Dict
import numpy as np
from utility.logger import logger


class SmartCache:
    """Intelligent caching system for RAG operations"""

    def __init__(self, cache_dir: str = "cache", ttl_seconds: int = 3600):
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories too
        except Exception as e:
            logger.warning(f"Failed to create cache directory {cache_dir}: {e}")
            # Fallback to a simple cache directory in current path
            self.cache_dir = Path("temp_cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ttl_seconds = ttl_seconds
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                return json.loads(self.metadata_file.read_text())
            except:
                logger.warning("Cache metadata corrupted, starting fresh")
        return {}

    def _save_metadata(self):
        """Save cache metadata"""
        self.metadata_file.write_text(json.dumps(self.metadata, indent=2))

    def _hash_key(self, key: str) -> str:
        """Create hash for cache key"""
        return hashlib.md5(key.encode()).hexdigest()

    def _is_expired(self, cache_key: str) -> bool:
        """Check if cache entry is expired"""
        if cache_key not in self.metadata:
            return True
        created_time = self.metadata[cache_key].get('created', 0)
        return time.time() - created_time > self.ttl_seconds

    def get(self, key: str, default=None) -> Any:
        """Get value from cache"""
        cache_key = self._hash_key(key)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists() or self._is_expired(cache_key):
            logger.debug(f"Cache miss: {key[:50]}...")
            return default

        try:
            with open(cache_file, 'rb') as f:
                value = pickle.load(f)
            logger.debug(f"Cache hit: {key[:50]}...")
            return value
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return default

    def set(self, key: str, value: Any):
        """Set value in cache"""
        cache_key = self._hash_key(key)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)

            self.metadata[cache_key] = {
                'original_key': key[:100],  # Store first 100 chars for debugging
                'created': time.time(),
                'size_bytes': cache_file.stat().st_size
            }
            self._save_metadata()
            logger.debug(f"Cached: {key[:50]}...")

        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self):
        """Clear all cache"""
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
        self.metadata = {}
        self._save_metadata()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_size = sum(item.get('size_bytes', 0) for item in self.metadata.values())
        return {
            'entries': len(self.metadata),
            'total_size_mb': total_size / 1024 / 1024,
            'cache_dir': str(self.cache_dir)
        }


class EmbeddingCache(SmartCache):
    """Specialized cache for embeddings"""

    def __init__(self, cache_dir: str = "cache/embeddings", ttl_seconds: int = 86400):  # 24 hour TTL
        super().__init__(cache_dir, ttl_seconds)

    def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        key = f"embedding:{model_name}:{text}"
        return self.get(key)

    def set_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        """Cache embedding"""
        key = f"embedding:{model_name}:{text}"
        self.set(key, embedding)


class QueryCache(SmartCache):
    """Specialized cache for query results"""

    def __init__(self, cache_dir: str = "cache/queries", ttl_seconds: int = 1800):  # 30 minute TTL
        super().__init__(cache_dir, ttl_seconds)

    def get_query_result(self, query: str, model_name: str, context_hash: str = "") -> Optional[Dict]:
        """Get cached query result"""
        key = f"query:{model_name}:{context_hash}:{query}"
        return self.get(key)

    def set_query_result(self, query: str, model_name: str, result: Dict, context_hash: str = ""):
        """Cache query result"""
        key = f"query:{model_name}:{context_hash}:{query}"
        self.set(key, result)


# Global cache instances with error handling
try:
    embedding_cache = EmbeddingCache()
    query_cache = QueryCache()
    general_cache = SmartCache()
    CACHE_INITIALIZED = True
except Exception as e:
    logger.warning(f"Failed to initialize caches: {e}")


    # Create dummy cache objects that don't actually cache
    class DummyCache:
        def get(self, key, default=None):
            return default

        def set(self, key, value):
            pass

        def clear(self):
            pass

        def get_stats(self):
            return {'entries': 0, 'total_size_mb': 0.0, 'cache_dir': 'disabled'}

        def get_embedding(self, text, model_name):
            return None

        def set_embedding(self, text, model_name, embedding):
            pass

        def get_query_result(self, query, model_name, context_hash=""):
            return None

        def set_query_result(self, query, model_name, result, context_hash=""):
            pass


    embedding_cache = DummyCache()
    query_cache = DummyCache()
    general_cache = DummyCache()
    CACHE_INITIALIZED = False


def cache_embedding(model_name: str):
    """Decorator to cache embedding results"""

    def decorator(func):
        def wrapper(text: str, *args, **kwargs):
            # Check cache first
            cached = embedding_cache.get_embedding(text, model_name)
            if cached is not None:
                return cached

            # Generate embedding
            result = func(text, *args, **kwargs)

            # Cache result
            if isinstance(result, np.ndarray):
                embedding_cache.set_embedding(text, model_name, result)

            return result

        return wrapper

    return decorator


def cache_query_result(model_name: str):
    """Decorator to cache query results"""

    def decorator(func):
        def wrapper(query: str, *args, **kwargs):
            # Create context hash from args (simplified)
            context_hash = hashlib.md5(str(args).encode()).hexdigest()[:8]

            # Check cache first
            cached = query_cache.get_query_result(query, model_name, context_hash)
            if cached is not None:
                return cached

            # Generate result
            result = func(query, *args, **kwargs)

            # Cache result
            if isinstance(result, dict) and 'answer' in result:
                query_cache.set_query_result(query, model_name, result, context_hash)

            return result

        return wrapper

    return decorator


def cache_stats():
    """Print cache statistics"""
    if not CACHE_INITIALIZED:
        print("\nCACHE STATISTICS")
        print("=" * 40)
        print("Cache system is disabled")
        return

    print("\nCACHE STATISTICS")
    print("=" * 40)

    embedding_stats = embedding_cache.get_stats()
    query_stats = query_cache.get_stats()
    general_stats = general_cache.get_stats()

    print(f"Embeddings: {embedding_stats['entries']} entries, {embedding_stats['total_size_mb']:.1f}MB")
    print(f"Queries: {query_stats['entries']} entries, {query_stats['total_size_mb']:.1f}MB")
    print(f"General: {general_stats['entries']} entries, {general_stats['total_size_mb']:.1f}MB")

    total_mb = embedding_stats['total_size_mb'] + query_stats['total_size_mb'] + general_stats['total_size_mb']
    print(f"Total Cache: {total_mb:.1f}MB")


def clear_all_caches():
    """Clear all caches"""
    if not CACHE_INITIALIZED:
        logger.info("Cache system is disabled, nothing to clear")
        return

    embedding_cache.clear()
    query_cache.clear()
    general_cache.clear()
    logger.info("All caches cleared")