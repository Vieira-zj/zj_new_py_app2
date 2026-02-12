import random
from typing import Dict, Optional, Tuple, Union

from rank_bm25 import BM25Okapi

# example: python


def test_py_typing():
    # union
    def get_number_v1() -> Union[int, None]:
        r = random.randint(1, 10)
        return r if r < 5 else None

    for _ in range(3):
        n = get_number_v1()
        if n:
            print("get number:", n)
        else:
            print("unknown")

    # optional
    def get_number_v2() -> Optional[int]:
        r = random.randint(1, 10)
        return r if r < 5 else None

    for _ in range(3):
        n = get_number_v2()
        if n:
            print("get number:", n)
        else:
            print("unknown")


def test_get_sublist():
    l = [1, 2, 3]
    # start / end can exceed len of list
    print(l[5:])
    print(l[:5])


def test_iter_enumerate():
    l = ["a", "b", "c", "x"]
    for i, _ in enumerate(l):
        print("index:", i)

    print("\nindex start from 1:")
    for i, _ in enumerate(l, 1):
        print("index:", i)


def test_iter_next():
    l = iter([1, 2, 3, 4])
    for _ in range(3):
        print(next(l))


# example: function


def test_fn_return_multi_param():
    def get_number_and_str() -> Tuple[int, str]:
        n = random.randint(1, 5)
        m: Dict[int, str] = {1: "one", 2: "two", 3: "three"}
        s = m.get(n, "nan")
        return n, s

    n, s = get_number_and_str()
    print(f"number={n}, string={s}")


# example: similarity 算法


def test_bm25_similarity():
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A quick brown dog outpaces a swift fox",
        "The dog is lazy but the fox is quick",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language",
    ]

    # simple whitespace tokenization
    tokenized_docs = [line.lower().split() for line in documents]
    bm25 = BM25Okapi(tokenized_docs)

    query = "quick fox"
    tokenized_query = query.lower().split()

    # get scores for all documents
    scores = bm25.get_scores(query=tokenized_query)
    print("scores:", [f"{score:.2f}" for score in scores])

    # get top N most relevant documents
    top_n = bm25.get_top_n(query=tokenized_query, documents=documents, n=3)
    print("\ntop 3 results:")
    for i, doc in enumerate(top_n, 1):
        print(f"  {i}. {doc}")


if __name__ == "__main__":
    # test_py_typing()

    test_get_sublist()
    # test_iter_enumerate()
    # test_iter_next()

    # test_fn_return_multi_param()

    # test_bm25_similarity()
