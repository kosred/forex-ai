import requests

urls = [
    "https://huggingface.co/datasets/Ehsanrs2/Forex_Factory_Calendar/resolve/main/Forex_Factory_Calendar.csv",
    "https://huggingface.co/datasets/Ehsanrs2/Forex_Factory_Calendar/resolve/master/Forex_Factory_Calendar.csv",
    "https://huggingface.co/datasets/Ehsanrs2/Forex_Factory_Calendar/resolve/main/train.csv",
    "https://huggingface.co/datasets/Ehsanrs2/Forex_Factory_Calendar/raw/main/Forex_Factory_Calendar.csv",
]

TIMEOUT = (5, 15)  # connect, read
HEADERS = {"User-Agent": "forex-ai-url-check/1.0"}


def check(url: str) -> None:
    """Check URL reachability with HEAD then GET fallback (streamed)."""
    with requests.Session() as session:
        session.headers.update(HEADERS)
        try:
            resp = session.head(url, allow_redirects=True, timeout=TIMEOUT)
            if resp.status_code >= 400 or resp.status_code in (403, 405):
                # Some CDNs/WAFs block HEAD — try a lightweight GET
                resp = session.get(url, allow_redirects=True, stream=True, timeout=TIMEOUT)
            print(f"{url}: {resp.status_code}")
        except Exception as exc:
            print(f"{url}: Error {exc}")


if __name__ == "__main__":
    for url in urls:
        check(url)
