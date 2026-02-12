from typing import List

from bs4 import BeautifulSoup, Tag


def test_parse_html_01():
    html_doc = """<html>
<head>
    <title>My Page</title>
</head>
<body>
    <h1>Welcome to my website!</h1>
    <p class="intro">This is an <span>introduction</span> paragraph.</p>
    <div id="main-content">
        <ul>
            <li>Item 1</li>
            <li class="highlight">Item 2</li>
            <li>Item 3</li>
        </ul>
        <a href="https://www.example.com" class="external-link">Visit Example.com</a>
    </div>
    <p>Another paragraph.</p>
</body>
</html>"""

    soup = BeautifulSoup(markup=html_doc, features="html.parser")
    if soup.title:
        print("--- Page Title ---")
        print(soup.title.string)
    if soup.h1:
        print("\n--- First H1 Tag ---")
        print(soup.h1.string)

    print("\n--- Paragraph with class 'intro' ---")
    intro_paragraph = soup.find("p", class_="intro")
    if intro_paragraph:
        print(intro_paragraph.get_text())

    print("\n--- Anchor Tag (Link) ---")
    link_tag = soup.find("a")
    if link_tag:
        print(f"Link Text: {link_tag.get_text()}")
        print(f"Link URL: {link_tag['href']}")
        # get() is safer for attributes that might not exist
        print(f"Link Class: {link_tag.get('class')}")

    print("\n--- Div with id 'main-content' ---")
    main_content_div = soup.find(id="main-content")
    if main_content_div:
        print(main_content_div.prettify())

    print("\n--- All List Items ---")
    if main_content_div:
        li_tags = main_content_div.find_all("li")
        for li_tag in li_tags:
            print(li_tag.get_text())


def test_parse_html_02():
    """convert a html table to markdown table."""
    html_doc = """<!doctype html>
<html>
  <head>
    <title>My Page</title>
  </head>
  <body>
    <h1>Welcome to my website!</h1>
    <p class="intro">This is an <span>introduction</span> paragraph.</p>
    <div>
      <table id="user_list">
        <thead>
          <tr>
            <th>User ID</th>
            <th>User Name</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>id1001</td>
            <td>Foo</td>
          </tr>
          <tr>
            <td>id1002</td>
            <td>Bar</td>
          </tr>
        </tbody>
      </table>
    </div>
  </body>
</html>"""

    soup = BeautifulSoup(markup=html_doc, features="html.parser")
    user_list = soup.find("table", id="user_list")
    if not user_list:
        print("user list table is not found")
        return

    print("html table:")
    print(user_list.prettify())

    print("\nmarkdown table:")
    print_html_table_as_markdown(user_list)


def print_html_table_as_markdown(tab_element: Tag):
    # parse table headers
    headers: List[str] = []
    splits: List[str] = []
    ths = tab_element.find_all("th")
    for th in ths:
        column = th.get_text()
        headers.append(column)
        splits.append("-" * len(column))

    # parse table rows
    rows: List[List[str]] = []
    tds = tab_element.find_all("td")
    for i in range(0, len(tds), len(ths)):
        rows.append([tds[i].get_text(), tds[i + 1].get_text()])

    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join(splits)} |")
    for row in rows:
        print(f"| {' | '.join(row)} |")


def test_parse_html_03():
    """walk through every element in the html doc tree."""
    html_doc = """<!doctype html>
<html>
  <head>
    <title>My Page</title>
  </head>
  <body>
    <h1>Welcome to my website!</h1>
    <p class="intro">This is an <span>introduction</span> paragraph.</p>
    <div>
      <table id="user_list">
        <thead>
          <tr>
            <th>User ID</th>
            <th>User Name</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>id1001</td>
            <td>Foo</td>
          </tr>
          <tr>
            <td>id1002</td>
            <td>Bar</td>
          </tr>
        </tbody>
      </table>
    </div>
  </body>
</html>"""

    soup = BeautifulSoup(markup=html_doc, features="html.parser")
    for element in soup.descendants:
        if not isinstance(element, Tag):
            continue

        match element.name:
            case "h1":
                print("h1 text:", element.get_text())
            case "p":
                print("paragraph text:", element.get_text())
            case "table":
                print("\nhtml table:")
                print(element.prettify())
                print("\nmarkdown table:")
                print_html_table_as_markdown(element)


if __name__ == "__main__":
    # test_parse_html_01()
    # test_parse_html_02()
    test_parse_html_03()
