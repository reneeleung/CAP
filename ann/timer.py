import json
import os
import threading
from filelock import FileLock

class TimerStorage:
    FILE_PATH = 'timer_general.json'
    FILE_LOCK_PATH = FILE_PATH + '.lock'
    CACHE_LOCK = threading.Lock()
    cache = {}

    @staticmethod
    def load_states(data_dir):
        # Acquire an inter-process lock before accessing the file.
        file_lock_path = os.path.join(data_dir, TimerStorage.FILE_LOCK_PATH)
        file_path = os.path.join(data_dir, TimerStorage.FILE_PATH)
        with FileLock(file_lock_path, timeout=10):
            if not os.path.exists(file_path):
                TimerStorage.cache = {}
                return
            try:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        TimerStorage.cache = {}
                        return
                    TimerStorage.cache = json.loads(content)
            except json.JSONDecodeError:
                print('Error: Invalid JSON format in timer_states.json', flush=True)

    @staticmethod
    def save_states(data_dir):
        file_lock_path = os.path.join(data_dir, TimerStorage.FILE_LOCK_PATH)
        file_path = os.path.join(data_dir, TimerStorage.FILE_PATH)
        # Acquire an inter-process lock to protect file write operations.
        with FileLock(file_lock_path, timeout=10):
            with open(file_path, 'w') as f:
                json.dump(TimerStorage.cache, f, indent=4)
                f.flush()  # Make sure the data is written immediately
                os.fsync(f.fileno())  # Force the OS to flush to disk

    @staticmethod
    def save_timer_state(file, elapsed_time, data_dir):
        if file is None:
            return {'status': 'failed'}
        with TimerStorage.CACHE_LOCK:
            TimerStorage.load_states(data_dir)
            TimerStorage.cache[file] = {'elapsedTime': elapsed_time}
            TimerStorage.save_states(data_dir)
        return {'status': 'saved'}

    @staticmethod
    def get_timer_state(file, data_dir):
        with TimerStorage.CACHE_LOCK:
            TimerStorage.load_states(data_dir)
            state = TimerStorage.cache.get(file, {'elapsedTime': 0})
        return state
