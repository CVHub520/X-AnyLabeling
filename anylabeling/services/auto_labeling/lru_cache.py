"""Thread-safe LRU cache implementation."""
from collections import OrderedDict
import threading


class LRUCache:
    """Thread-safe LRU cache implementation."""

    def __init__(self, maxsize=10):
        self.maxsize = maxsize
        self.lock = threading.Lock()
        self._cache = OrderedDict()

    def get(self, key):
        """Get value from cache. Returns None if key is not present."""
        with self.lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key, value):
        """Put value into cache. If cache is full, oldest item is evicted."""
        with self.lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def find(self, key):
        """Returns True if key is in cache, False otherwise."""
        with self.lock:
            return key in self._cache
