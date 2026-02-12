import heapq
import threading
from dataclasses import dataclass, field
from typing import Any, Optional, Self

# example: singleton


class SingletonCls:
    _instance: Optional[Self] = None
    _lock = threading.Lock()

    def __new__(cls) -> Self:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, data: Any):
        if self._initialized:  # pylint: disable=E0203:access-member-before-definition
            return
        self.data = data
        self._initialized = True

    def get_data(self) -> Any:
        return self.data


# example: priority queue


def test_priority_queue():
    @dataclass(order=True)
    class Task:
        name: str = field(compare=False)  # exclude from comparison
        priority_id: int

    pq = []
    heapq.heappush(pq, Task(priority_id=2, name="Write report"))
    heapq.heappush(pq, Task(priority_id=1, name="Fix critical bug"))
    heapq.heappush(pq, Task(priority_id=3, name="Review code"))
    heapq.heappush(pq, Task(priority_id=1, name="Deploy hotfix"))

    print("process tasks:")
    while pq:
        task = heapq.heappop(pq)
        print(f"  [{task.priority_id}]: {task.name}")


if __name__ == "__main__":
    test_priority_queue()
