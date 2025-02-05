import aiosqlite
import logging
import json
import time
from typing import Any, Optional

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DB_PATH = 'cache.db'
MAX_CACHE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_ENTRIES = 10000
EVICTION_BATCH = 100

async def init_db():
    """Initialize the SQLite database with required tables."""
    try:
        async with aiosqlite.connect(CACHE_DB_PATH) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    last_accessed INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                )
            ''')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache(last_accessed)')
            await db.commit()
    except Exception as e:
        logger.error(f"Error initializing cache database: {e}")
        raise

async def get_cache(key: str) -> Optional[Any]:
    """
    Retrieve a value from the cache by key.
    
    Args:
        key: The cache key to lookup
        
    Returns:
        The cached value if found, None otherwise
    """
    try:
        async with aiosqlite.connect(CACHE_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            
            # Update last accessed time and retrieve value
            current_time = int(time.time())
            async with db.execute(
                'SELECT value FROM cache WHERE key = ?',
                (key,)
            ) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    # Update last_accessed time
                    await db.execute(
                        'UPDATE cache SET last_accessed = ? WHERE key = ?',
                        (current_time, key)
                    )
                    await db.commit()
                    
                    try:
                        return json.loads(row['value'])
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding cached value for key {key}")
                        return None
                
                return None
                
    except Exception as e:
        logger.error(f"Error retrieving from cache: {e}")
        return None

async def set_cache(key: str, value: Any) -> bool:
    """
    Store a value in the cache with the given key.
    
    Args:
        key: The cache key
        value: The value to cache
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Serialize value to JSON
        serialized_value = json.dumps(value)
        value_size = len(serialized_value.encode('utf-8'))
        current_time = int(time.time())
        
        async with aiosqlite.connect(CACHE_DB_PATH) as db:
            # Check current cache size
            async with db.execute('SELECT SUM(size) as total_size FROM cache') as cursor:
                row = await cursor.fetchone()
                current_size = row[0] or 0
                
            # Perform LRU eviction if needed
            if current_size + value_size > MAX_CACHE_SIZE:
                await evict_lru_entries(db, value_size)
            
            # Insert or update cache entry
            await db.execute('''
                INSERT OR REPLACE INTO cache (key, value, size, last_accessed, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (key, serialized_value, value_size, current_time, current_time))
            
            await db.commit()
            return True
            
    except Exception as e:
        logger.error(f"Error setting cache value: {e}")
        return False

async def evict_lru_entries(db, required_space: int):
    """
    Evict least recently used entries to free up required space.
    
    Args:
        db: Database connection
        required_space: Amount of space needed in bytes
    """
    try:
        while True:
            # Get total cache size
            async with db.execute('SELECT SUM(size) as total_size FROM cache') as cursor:
                row = await cursor.fetchone()
                current_size = row[0] or 0
                
            if current_size + required_space <= MAX_CACHE_SIZE:
                break
                
            # Delete batch of oldest entries
            async with db.execute('''
                DELETE FROM cache 
                WHERE key IN (
                    SELECT key FROM cache 
                    ORDER BY last_accessed ASC 
                    LIMIT ?
                )
            ''', (EVICTION_BATCH,)) as cursor:
                if cursor.rowcount == 0:
                    break
                    
            await db.commit()
            
    except Exception as e:
        logger.error(f"Error during cache eviction: {e}")
        raise

# Add the following initialization function:
async def init_persistent_cache():
    return await init_db()