import argparse
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from typing import Any

XMIND_NS = {
    "xmind": "urn:xmind:xmap:xmlns:content:2.0",
}


def _parse_topic(topic_el: ET.Element) -> dict:
    """Recursively parse an XML <topic> element into a dict."""
    node = {}

    # Title
    title_el = topic_el.find("xmind:title", XMIND_NS)
    if title_el is not None and title_el.text:
        node["title"] = title_el.text

    # Notes (plain text)
    notes_el = topic_el.find(".//xmind:notes/xmind:plain", XMIND_NS)
    if notes_el is not None and notes_el.text:
        node["notes"] = notes_el.text

    # Labels
    labels = [
        lbl.text
        for lbl in topic_el.findall("xmind:labels/xmind:label", XMIND_NS)
        if lbl.text
    ]
    if labels:
        node["labels"] = labels

    # Children (subtopics)
    children_el = topic_el.find("xmind:children/xmind:topics", XMIND_NS)
    if children_el is not None:
        children = [
            _parse_topic(t) for t in children_el.findall("xmind:topic", XMIND_NS)
        ]
        if children:
            node["children"] = children

    return node


def parse_xml_content(xml_bytes: bytes) -> Any:
    """Parse the legacy XML content.xml into a list of sheets."""
    root = ET.fromstring(xml_bytes)

    sheets = []
    for sheet_el in root.findall("xmind:sheet", XMIND_NS):
        sheet = {}
        title_el = sheet_el.find("xmind:title", XMIND_NS)
        if title_el is not None:
            sheet["title"] = title_el.text

        root_topic = sheet_el.find("xmind:topic", XMIND_NS)
        if root_topic is not None:
            sheet["rootTopic"] = _parse_topic(root_topic)

        sheets.append(sheet)

    return sheets


def xmind_to_json(xmind_path: str) -> Any:
    """
    Open an .xmind file and return its content as a Python object (list of sheets).
    """
    with zipfile.ZipFile(xmind_path, mode="r") as zf:
        names = zf.namelist()

        # Prefer the JSON format (XMind Zen / XMind 8+)
        if "content.json" in names:
            raw = zf.read("content.json")
            return json.loads(raw)

        # Fall back to XML format (older XMind)
        if "content.xml" in names:
            raw = zf.read("content.xml")
            return parse_xml_content(raw)

        raise ValueError(
            f"Not a valid XMind file - no content.json or content.xml found.\n"
            f"Archive contains: {names}"
        )


def main():
    parser = argparse.ArgumentParser(description="Convert an XMind file to JSON.")
    parser.add_argument("xmind_file", type=str, help="Path to the .xmind file")
    parser.add_argument(
        "-o", "--output", type=str, help="Output JSON file (default: stdout)"
    )
    parser.add_argument(
        "--indent", type=int, default=2, help="JSON indent level (default: 2)"
    )

    args = parser.parse_args()
    try:
        data = xmind_to_json(args.xmind_file)
    except (zipfile.BadZipFile, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexcpected exception: {e}")
        sys.exit(1)

    json_str = json.dumps(data, indent=args.indent, ensure_ascii=False)

    if args.output:
        with open(args.output, mode="w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"Dump to {args.output}")
    else:
        print("Xmind json:")
        print(json_str)


if __name__ == "__main__":
    # cli:
    # cd apps/xmind_to_json
    # uv run main.py input.xmind
    # uv run main.py input.xmind -o output.json

    main()
