# -- coding: utf-8 --
import json

import jsonpath_ng as jp
import jsonpath_ng.ext as jpx


def test_jsonpath_01():
    raw_data = """
{
    "name": "John",
    "age": 30,
    "place of residence": "New York"
}
"""
    json_obj = json.loads(raw_data)
    results = jp.parse("$.name").find(json_obj)
    print(results[0].value)


def test_jsonpath_02():
    data = """
{
    "cities": [
        {
            "name": "Trenton",
            "state": "New Jersey",
            "residents": 90048,
            "iscapital": true,
            "neighborhood Central West": {
                "residents": 1394    
            }
        },
        {
            "name": "Hamburg",
            "state": "Hamburg",
            "residents": 1841000,
            "iscapital": false
        },
        {
            "name": "New York City",
            "state": "New York",
            "residents ": 8804190,
            "iscapital": false
        },
        {
            "name": "Los Angeles",
            "state": "California",
            "residents": 3898767
        }
    ]
}
"""
    json_data = json.loads(data)

    # $ => Root-object. e.g. $.marcus.age
    # [x] => Element in an array. e.g. $.people[5]
    print("Names of all cities:")
    query = jp.parse("$.cities[*].name")
    for match in query.find(json_data):
        print(match.value)

    # .. => Recursive search for a field. Fields in sub-objects are also searched.
    print("\nAll fields labeled [residents]")
    query = jp.parse("$.cities..residents")
    match = query.find(json_data)
    for i in match:
        print(i.value)

    # @ => Object currently being searched, e.g. $.people[?(@.age < 50)]
    # .[?(filter)] => General syntax for array filters. e.g. $.people[?(@.name == "Anne")]
    print("\nThe names of all cities that are not called [Trenton]:")
    query = jpx.parse('$.cities[?(@.name != "Trenton")].name')
    for match in query.find(json_data):
        print(match.value)

    print("\nNames of all cities with less than 1.5 million residents:")
    query = jpx.parse("$.cities[?@.residents < 1500000].name")
    for match in query.find(json_data):
        print(match.value)

    # e.g. $.people[?(@.place of residence == Newark & @.age > 40)]


if __name__ == "__main__":
    # test_jsonpath_01()
    test_jsonpath_02()

    print("jsonpath test finished")
