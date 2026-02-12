from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


def get_page_links(url: str):
    resp = requests.get(url, timeout=3)
    soup = BeautifulSoup(resp.text, "html.parser")

    links = soup.find_all("a")
    for link in links:
        print("link text:", link.text.strip())
        print("url:", urljoin(url, str(link.get("href"))))


def get_page_images(url: str):
    resp = requests.get(url, timeout=3)
    soup = BeautifulSoup(resp.text, "html.parser")

    imgs = soup.find_all("img")
    for img in imgs:
        print("image uri:", urljoin(url, str(img.get("src"))))
        print("image desc:", img.get("alt"))


def save_image(img_url: str, img_name: str):
    img_data = requests.get(img_url, timeout=3).content
    save_path = f"images/{img_name}.jpg"
    with open(save_path, mode="wb") as f:
        f.write(img_data)
    print(f"saved {img_url} to {save_path}")


if __name__ == "__main__":
    pass
