# -- coding: utf-8 --
import dataclasses
import json
import random
import re

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Final, List, Optional, TypedDict, Union

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

# example: base


def test_import_lib():
    # pylint: disable=C0415:import-outside-toplevel
    from utils import base

    base.hello_world()


def test_cal_division():
    float_res = 10 / 3
    print(f"result: {float_res:.2f}")

    int_res = 10 // 3  # 返回整数
    print("result:", int_res)


def test_str_padding():
    s = "hello"
    print(f"|{s.ljust(10)}|")
    print(f"|{s.rjust(10)}|")

    print("\nadjust fields:")
    fields = ["ab", "xyz", "longword", "hello"]
    max_len = max(len(field) for field in fields)
    ljust_fields = [col.ljust(max_len) for col in fields]
    for field in ljust_fields:
        print(f"|{field}|")


def test_var_ops():
    main_mode: Final[str] = "main"
    print("mode:", main_mode)


def test_set_ops():
    s1: set[str] = set(["a", "b", "c"])
    s2: set[str] = set(["d", "b", "c"])
    print("diff:", s1 - s2)
    print("merged:", s1.intersection(s2))


def test_dict_ops():
    d = {
        "5": "five",
        "2": "two",
        "1": "one",
        "8": "eight",
        "4": "four",
        "10": "ten",
    }
    # iterate in order
    print("raw dict:")
    for item in d.items():
        print(item)

    # get 1st key, and pop item
    k = next(iter(d))
    d.pop(k)

    # update value
    d.update({"10": "Ten", "8": "Eight"})

    print("\nafter updated:")
    for k, v in d.items():
        print(f"{k}={v}")


def test_default_dict():
    counts = defaultdict(int)
    for c in "hello world, wowl":
        if c.strip():
            counts[c] += 1

    for k, v in counts.items():
        print(f"{k}:{v}")


def test_counter():
    chars: list[str] = []
    for c in "hello world":
        if c.strip():
            chars.append(c)

    counter = Counter(chars)
    counter.update(" ,wowl")
    for k, v in counter.items():
        print(f"{k}:{v}")

    result = counter.most_common(1)
    if result[0]:
        k, v = result[0]
        print(f"\nmost common: {k}:{v}")


def test_sorted():
    # list
    l = ["ab", "abc", "a"]
    print("sorted:", sorted(l, key=len))

    # dict
    d = {
        "five": "5",
        "two": "2",
        "one": "1",
        "eight": "8",
    }
    result = sorted(d.items(), key=lambda item: item[1])
    print("sorted dict:", result)


def test_catch_json_err():
    s = '{"data":{"message":"ok"}'
    try:
        d = json.loads(s)
        print("load:", d)
    except json.JSONDecodeError as e:
        print(f"json decode error: {e}")
    except Exception as e:
        print(f"unexpected error: {e}")


# example: typed dict


def test_class_with_typeddict():
    class Employee(TypedDict):
        name: str
        id: int
        department: str
        is_active: bool

    # emp: Employee = {
    #     "name": "Bar",
    #     "id": 101,
    #     "department": "Engineering",
    #     "is_active": True,
    # }

    emp = Employee(name="Bar", id=101, department="Engineering", is_active=True)

    print(f"type: {type(emp)}")
    print(f"is dict: {isinstance(emp, dict)}")
    print()

    print(f"name: {emp['name']}")
    print(f"id: {emp['id']}")
    print(f"department: {emp['department']}")
    print(f"active: {'Yes' if emp.get('is_active') else 'No'}")

    print(f"\nemployee: {json.dumps(emp)}")


# example: dataclass
# parameters: slots=True, frozen=True, kw_only=True, order=True


@dataclass
class User:
    user_id: int
    username: str
    email: str

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(slots=True, frozen=True, kw_only=True)
class Address:
    street: str
    city: str
    zip_code: str


@dataclass(slots=True, frozen=True, kw_only=True)
class Person:
    name: str
    age: int
    email: Optional[str] = None
    addresses: List[Address] = dataclasses.field(default_factory=list)
    is_active: bool = True


def test_datacls_and_dict():
    addr1 = Address(street="123 Main St", city="Anytown", zip_code="12345")
    addr2 = Address(street="456 Oak Ave", city="Otherville", zip_code="67890")

    person1 = Person(
        name="Foo",
        age=30,
        email="foo@example.com",
        addresses=[addr1, addr2],
    )
    person2 = Person(
        name="Bar",
        age=25,
        is_active=False,
    )

    # dataclass -> dict
    dict1 = dataclasses.asdict(person1)
    dict2 = dataclasses.asdict(person2)
    print("person1 dict:", dict1)
    print("person2 dict:", dict2)

    # dict to dataclass
    print()
    person3 = Person(**dict1)
    print("person3:", person3.name, person3.email, person3.addresses)
    person4 = Person(**dict2)
    print("person4:", person4.name, person4.age)


# example: expr


def replace_date(text: str, target_date: str) -> str:
    # replace all dates with target_date
    pattern = r"\d{4}-\d{2}-\d{2}"
    return re.sub(pattern, target_date, text)


def test_expr_replace_date():
    s = "created 2024-10-01, updated 2025-05-18"
    new_str = replace_date(s, "2026-03-11")
    print("result:", new_str)


# example: datetime


def get_next_friday(input_date: str) -> str:
    dt = datetime.strptime(input_date, "%Y-%m-%d")
    for _ in range(7):
        if dt.weekday() + 1 == 5:
            return dt.strftime("%Y-%m-%d")
        dt += timedelta(days=1)
    raise ValueError("not go here")


def test_get_next_friday():
    result = get_next_friday("2026-06-26")
    print("next friday:", result)


def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5


def get_2nd_last_working_day(input_date: str) -> str:
    dt = datetime.strptime(input_date, "%Y-%m-%d")
    next_month = datetime.strptime(f"{dt.year}-{dt.month+1:02d}-01", "%Y-%m-%d")
    result = next_month - timedelta(days=2)

    for _ in range(2):
        if is_weekend(result):
            result -= timedelta(days=1)
    return result.strftime("%Y-%m-%d")


def test_get_2nd_last_working_day():
    result = get_2nd_last_working_day("2025-03-14")
    print("result:", result)


# example: retry


def my_get_number() -> Union[int, str]:
    n = random.randint(1, 10)
    print("return number:", n)
    return n if n < 8 else str(n)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),  # seconds
    # wait=wait_exponential(multiplier=1, min=2, max=10) # 2s, 4s, 8s, 10s, 10s ...
    retry=retry_if_exception_type(ValueError),
)
def my_call(get_number: Callable):
    number = get_number()
    if isinstance(number, int):
        if number < 3:
            print("number:", number)
            return

        msg = f"invalid number: {number}"
        print(msg)
        raise ValueError(msg)

    msg = "input must be int"
    print(msg)
    raise TypeError(msg)


def test_my_call_with_retry():
    my_call(my_get_number)
    print("run finished")


if __name__ == "__main__":
    # test_import_lib()

    # test_cal_division()
    # test_str_padding()

    # test_var_ops()
    # test_set_ops()
    # test_dict_ops()

    # test_default_dict()
    # test_counter()

    # test_sorted()
    # test_catch_json_err()

    # test_class_with_typeddict()
    # test_datacls_and_dict()

    test_expr_replace_date()

    # test_get_next_friday()
    # test_get_2nd_last_working_day()

    # test_my_call_with_retry()
