# -- coding: utf-8 --
import dataclasses
import random
from dataclasses import dataclass
from typing import Callable, Final, List, Optional, TypedDict, Union

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

# example: base


def test_var_ops():
    main_mode: Final[str] = "main"
    print("mode:", main_mode)


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
    for item in d.items():
        print(item)

    # get 1st key, and pop item
    k = next(iter(d))
    d.pop(k)

    print("\nafter pop:")
    for item in d.items():
        print(item)


def test_cal_division():
    float_res = 10 / 3
    print(f"result: {float_res:.2f}")

    int_res = 10 // 3  # 返回整数
    print("result:", int_res)


# example: typed dict


def test_class_with_typeddict():
    class Employee(TypedDict):
        name: str
        id: int
        department: str
        is_active: bool

    emp: Employee = {
        "name": "Bar",
        "id": 1024,
        "department": "Engineering",
        "is_active": True,
    }

    print(f"type: {type(emp)}")
    print(f"is dict: {isinstance(emp, dict)}")
    print()

    print(f"name: {emp['name']}")
    print(f"id: {emp['id']}")
    print(f"department: {emp['department']}")
    print(f"active: {'Yes' if emp.get('is_active') else 'No'}")


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
    # test_var_ops()
    test_dict_ops()
    # test_cal_division()

    # test_class_with_typeddict()
    # test_datacls_and_dict()

    # test_my_call_with_retry()
