import PyPDF2


def test_pdf_meta_read():
    reader = PyPDF2.PdfReader("/tmp/test/out.pdf")

    print(f"total pages: {len(reader.pages)}")

    metadata = reader.metadata
    if metadata:
        print(f"title: {metadata.title}")
        print(f"author: {metadata.author}")


def test_pdf_content_read():
    reader = PyPDF2.PdfReader("/tmp/test/out.pdf")
    if len(reader.pages) == 0:
        print("no page found in pdf")
        return

    page = reader.pages[0]
    text = page.extract_text()  # pylint: disable=E1101:no-member
    print(f"\npdf content:\n{text}")


def test_merge_pdf_files():
    writer = PyPDF2.PdfWriter()

    pdf_files = ["file1.pdf", "file2.pdf", "file3.pdf"]
    for pdf_file in pdf_files:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            writer.add_page(page)

    with open("merged.pdf", mode="wb") as out_file:
        writer.write(out_file)


if __name__ == "__main__":
    print("py pdf version:", PyPDF2.__version__)

    # test_pdf_meta_read()
    test_pdf_content_read()
