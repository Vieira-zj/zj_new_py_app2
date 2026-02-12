import heapq
from dataclasses import dataclass, field

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
