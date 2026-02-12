import asyncio
import time
from concurrent.futures import ThreadPoolExecutor


def my_download_file(name: str, duration: float):
    print(f"start download {name}, take {duration} seconds ...")
    time.sleep(duration)
    print(f"download {name} finished")
    return name


async def my_async_download_file(name: str, duration: float):
    print(f"start download {name}, take {duration} seconds ...")
    await asyncio.sleep(duration)
    print(f"download {name} finished")
    return name


# example: thread pool executor


def test_pool_executor():
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i in range(1, 8):
            name, duration = f"file_{i}", i / 2
            future = executor.submit(my_download_file, name, duration)
            futures.append(future)

        results = []
        for i, future in enumerate(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"unexpected error in task {i}: {e}")

        print(f"\nall download tasks finished, and results:\n{results}")
        print(f"take: {(time.perf_counter() - start_time):.2f} seconds")


# example: asyncio


async def test_asyncio_pool_executor():
    start_time = time.perf_counter()
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i in range(1, 8):
            name, duration = f"file_{i}", i / 2
            future = loop.run_in_executor(executor, my_download_file, name, duration)
            futures.append(future)

        # here, future can be await
        results = await asyncio.gather(*futures)
        print(f"\nall download tasks finished, and results:\n{results}")
        print(f"take: {(time.perf_counter() - start_time):.2f} seconds")


async def test_asyncio_gather():
    start_time = time.perf_counter()

    task1 = my_async_download_file("fileA.txt", 2)
    task2 = my_async_download_file("fileB.json", 1)
    task3 = my_async_download_file("fileC.yml", 1.5)

    print("start run download tasks parallel ...")
    results = await asyncio.gather(task1, task2, task3)

    print(f"\nall download tasks finished, and results:\n{results}")
    print(f"take: {(time.perf_counter() - start_time):.2f} seconds")


async def test_asyncio_task_group():
    start_time = time.perf_counter()

    task1 = my_async_download_file("fileA.txt", 2)
    task2 = my_async_download_file("fileB.json", 1)
    task3 = my_async_download_file("fileC.yml", 1.5)

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(task) for task in (task1, task2, task3)]

    # task group will wait for all tasks to complete before exiting context, so no need to put await
    results = [task.result() for task in tasks]
    print(f"\nall download tasks finished, and results:\n{results}")
    print(f"take: {(time.perf_counter() - start_time):.2f} seconds")


if __name__ == "__main__":
    test_pool_executor()
    # asyncio.run(test_asyncio_pool_executor())

    # asyncio.run(test_asyncio_gather())
    # asyncio.run(test_asyncio_task_group())
