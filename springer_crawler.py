"""
Springer Journal Article Crawler
Downloads articles from https://link.springer.com/journal/{ID}/articles?filter-by-volume={volume}
- PDFs saved by article title
- BibTeX files saved per article
- Titles and abstracts stored in JSON

Output directory structure:
  papers/
    └── {journal_name}{year}(volume)/
        ├── pdfs/
        ├── bibs/
        └── articles.json
"""

import argparse
import re
import json
import time
import logging
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL = "https://link.springer.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
DELAY = 2
PAPERS_DIR = Path("papers")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def sanitize_filename(name: str, max_len: int = 150) -> str:
    name = re.sub(r"\\[\\^_{}()$]", "", name)
    name = re.sub(r"\^[a-zA-Z0-9]+", "", name)
    name = re.sub(r'[^a-zA-Z0-9 \-,.;&]', " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name.rstrip(". ")
    return name[:max_len]


def sanitize_dirname(name: str) -> str:
    """Clean journal name for compact directory naming."""
    # Remove common words and clean
    name = re.sub(r"\b(The|Journal|Nature|Springer)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^a-zA-Z0-9]", "", name)
    name = name.strip()
    return name


def get_output_dir(journal_id: str, volume: str, journal_name: str | None = None, year: str | None = None) -> Path:
    """Format output dir as {journal_name}{year}(volume)."""
    if journal_name and year:
        clean_journal = sanitize_dirname(journal_name)
        return PAPERS_DIR / f"{clean_journal}{year}(volume{volume})"
    return PAPERS_DIR / f"journal_{journal_id}_v{volume}"


# ── BibTeX helpers ──────────────────────────────────────────────────────────

def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def generate_bib_key(authors: list[str], year: str, articleno: str) -> str:
    if not authors:
        return f"unknown{year}_{articleno}"
    first = authors[0]
    parts = first.split(",")
    last_name = parts[0].strip()
    last_name = strip_accents(last_name)
    last_name = re.sub(r"[^a-zA-Z]", "", last_name)
    return f"{last_name}{year}_{articleno}"


def format_bibtex_authors(authors: list[str]) -> str:
    formatted = []
    for a in authors:
        a = a.strip().rstrip(".")
        parts = a.split(",", 1)
        if len(parts) == 2:
            last = parts[0].strip()
            first = parts[1].strip()
            formatted.append(f"{last}, {first}")
        else:
            formatted.append(a)
    return " and ".join(formatted)


def clean_bib_title(title: str) -> str:
    title = re.sub(r"\\\([^\)]*?\)", "", title)
    title = re.sub(r"\^[a-zA-Z0-9]+", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    title = title.replace("&", r"\&")
    return title


def to_bibtex(detail: dict) -> str:
    authors = detail.get("authors", [])
    year = (detail.get("pub_date") or "")[:4] or "2025"
    articleno = detail.get("articleno", "")

    key = generate_bib_key(authors, year, articleno)
    author_str = format_bibtex_authors(authors)
    title = clean_bib_title(detail.get("title", ""))
    journal = detail.get("journal", "")
    volume = detail.get("volume", "")
    doi = detail.get("doi", "")

    lines = [
        f"@article{{{key},",
        f"  author    = {{{author_str}}},",
        f"  title     = {{{{{title}}}}},",
        f"  journal   = {{{journal}}},",
    ]
    if volume:
        lines.append(f"  volume    = {{{volume}}},")
    if articleno:
        lines.append(f"  number    = {{{articleno}}},")
    if year:
        lines.append(f"  year      = {{{year}}},")
    if doi:
        lines.append(f"  doi       = {{{doi}}},")
    lines.append("}")
    return "\n".join(lines)


def load_browser_cookies(browser: str = "chrome") -> dict:
    """Load cookies for springer.com from the specified browser (default: chrome)."""
    try:
        import browser_cookie3
    except ImportError:
        log.error("browser-cookie3 not installed. Run: pip install browser-cookie3")
        return {}

    cookie_jar = None
    browser = browser.lower()
    try:
        if browser == "chrome":
            cookie_jar = browser_cookie3.chrome(domain_name="springer.com")
        elif browser == "firefox":
            cookie_jar = browser_cookie3.firefox(domain_name="springer.com")
        elif browser == "edge":
            cookie_jar = browser_cookie3.edge(domain_name="springer.com")
        elif browser == "opera":
            cookie_jar = browser_cookie3.opera(domain_name="springer.com")
        else:
            log.error("Unknown browser: %s. Use chrome/firefox/edge/opera.", browser)
            return {}
    except Exception as e:
        log.error("Failed to load cookies from %s: %s", browser, e)
        log.info("Make sure you have logged into Springer via your institution in %s.", browser)
        return {}

    cookies = {}
    if cookie_jar:
        for c in cookie_jar:
            cookies[c.name] = c.value
    log.info("Loaded %d cookies from %s for springer.com", len(cookies), browser)
    return cookies


def fetch(session: requests.Session, url: str) -> BeautifulSoup | None:
    for attempt in range(3):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            log.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)
            time.sleep(DELAY * (attempt + 1))
    return None


def collect_article_links(session: requests.Session, list_url: str) -> tuple[list[dict], str, str]:
    """
    Parse list pages and return article links along with journal name and year.
    Returns (articles, journal_name, year)
    """
    articles = []
    seen = set()
    page = 1
    journal_name = ""
    year = ""

    while True:
        url = list_url if page == 1 else f"{list_url}&page={page}"
        log.info("Fetching list page %d: %s", page, url)
        soup = fetch(session, url)
        if not soup:
            break

        # Extract journal name from page title
        if page == 1 and not journal_name:
            title_tag = soup.select_one("title")
            if title_tag:
                title_text = title_tag.get_text(strip=True)
                # Format: "Articles | The VLDB Journal | Springer Nature Link"
                parts = title_text.split("|")
                if len(parts) >= 2:
                    journal_name = parts[1].strip()

        # Extract year from meta or page content
        if page == 1 and not year:
            # Try citation meta
            for meta in soup.select('meta[name="citation_publication_date"]'):
                date_str = meta.get("content", "")
                if date_str:
                    year = date_str[:4]
                    break

        items = soup.select("article.app-card-open h2.app-card-open__heading a")
        if not items:
            break

        for a_tag in items:
            href = a_tag.get("href", "")
            title = a_tag.get_text(strip=True)
            if not href or href in seen:
                continue
            seen.add(href)
            articles.append({
                "doi": href.replace("/article/", ""),
                "title": title,
                "url": BASE_URL + href,
            })

        next_link = soup.select_one("li.eds-c-pagination__item--next a")
        if next_link and next_link.get("href"):
            page += 1
            time.sleep(DELAY)
        else:
            break

    log.info("Collected %d article links (journal: %s, year: %s)", len(articles), journal_name, year)
    return articles, journal_name, year


def parse_article_detail(session: requests.Session, article: dict) -> dict | None:
    soup = fetch(session, article["url"])
    if not soup:
        return None

    abstract = ""
    abs_section = soup.select_one("#Abs1-content p")
    if abs_section:
        abstract = abs_section.get_text(strip=True)

    pdf_url = ""
    meta_pdf = soup.select_one('meta[name="citation_pdf_url"]')
    if meta_pdf:
        pdf_url = meta_pdf.get("content", "")

    authors = [
        meta.get("content", "")
        for meta in soup.select('meta[name="citation_author"]')
    ]

    journal = ""
    meta_journal = soup.select_one('meta[name="citation_journal_title"]')
    if meta_journal:
        journal = meta_journal.get("content", "")

    volume = ""
    meta_vol = soup.select_one('meta[name="citation_volume"]')
    if meta_vol:
        volume = meta_vol.get("content", "")

    issue = ""
    meta_issue = soup.select_one('meta[name="citation_issue"]')
    if meta_issue:
        issue = meta_issue.get("content", "")

    pub_date = ""
    meta_date = soup.select_one('meta[name="citation_publication_date"]')
    if meta_date:
        pub_date = meta_date.get("content", "")

    # Article number
    articleno = ""
    meta_firstpage = soup.select_one('meta[name="citation_firstpage"]')
    if meta_firstpage:
        articleno = meta_firstpage.get("content", "")

    return {
        **article,
        "abstract": abstract,
        "pdf_url": pdf_url,
        "authors": authors,
        "journal": journal,
        "volume": volume,
        "issue": issue,
        "pub_date": pub_date,
        "articleno": articleno,
    }


def download_pdf(session: requests.Session, url: str, save_path: Path) -> bool:
    try:
        resp = session.get(
            url,
            headers={**HEADERS, "Accept": "application/pdf"},
            stream=True,
            timeout=60,
        )
        content_type = resp.headers.get("Content-Type", "")
        if resp.status_code != 200 or "pdf" not in content_type.lower():
            log.warning(
                "PDF not available for %s (status=%d, ct=%s)",
                url, resp.status_code, content_type,
            )
            return False
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size = save_path.stat().st_size
        if size < 1000:
            log.warning("PDF suspiciously small (%d bytes): %s", size, save_path)
            save_path.unlink(missing_ok=True)
            return False
        log.info("Downloaded PDF: %s (%d KB)", save_path.name, size // 1024)
        return True
    except requests.RequestException as e:
        log.error("PDF download failed for %s: %s", url, e)
        return False


def retry_missing_pdfs(session: requests.Session, output_dir: Path):
    """Re-download PDFs that failed previously, using existing articles.json."""
    json_path = output_dir / "articles.json"
    pdf_dir = output_dir / "pdfs"

    if not json_path.exists():
        log.error("No articles.json found. Run a full crawl first.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    missing = [a for a in articles if not a["pdf_saved"]]
    if not missing:
        log.info("All PDFs already downloaded!")
        return

    log.info("Retrying %d missing PDFs...", len(missing))
    for i, article in enumerate(missing, 1):
        log.info("[%d/%d] %s", i, len(missing), article["title"][:80])

        # Re-fetch article page to get fresh PDF URL
        article_url = BASE_URL + "/article/" + article["doi"]
        soup = fetch(session, article_url)
        if not soup:
            continue

        meta_pdf = soup.select_one('meta[name="citation_pdf_url"]')
        pdf_url = meta_pdf.get("content", "") if meta_pdf else ""
        if not pdf_url:
            log.warning("No PDF URL found for: %s", article["title"][:60])
            continue

        safe_title = sanitize_filename(article["title"])
        pdf_path = pdf_dir / f"{safe_title}.pdf"
        ok = download_pdf(session, pdf_url, pdf_path)

        # Update article record
        for a in articles:
            if a["doi"] == article["doi"]:
                a["pdf_saved"] = ok
                break

        time.sleep(DELAY)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    pdf_count = sum(1 for a in articles if a["pdf_saved"])
    log.info("Done! %d/%d PDFs downloaded.", pdf_count, len(articles))


def full_crawl(session: requests.Session, list_url: str, output_dir: Path):
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    (output_dir / "bibs").mkdir(parents=True, exist_ok=True)

    articles, journal_name, year = collect_article_links(session, list_url)
    if not articles:
        log.error("No articles found. Exiting.")
        return

    results = []
    total = len(articles)
    for i, article in enumerate(articles, 1):
        log.info("Processing %d/%d: %s", i, total, article["title"][:80])
        detail = parse_article_detail(session, article)
        if not detail:
            log.warning("Skipping article (detail fetch failed): %s", article["title"][:60])
            continue

        safe_title = sanitize_filename(detail["title"])

        # Save individual bib file
        bib_content = to_bibtex(detail)
        bib_key = generate_bib_key(
            detail.get("authors", []),
            (detail.get("pub_date") or "")[:4] or "2025",
            detail.get("articleno", ""),
        )
        bib_path = output_dir / "bibs" / f"{safe_title}.bib"
        bib_path.write_text(bib_content, encoding="utf-8")
        log.info("Saved bib: %s", bib_path.name)

        # Download PDF
        pdf_path = output_dir / "pdfs" / f"{safe_title}.pdf"
        if pdf_path.exists():
            log.info("PDF already exists, skipping: %s", pdf_path.name)
            detail["pdf_saved"] = True
        elif detail["pdf_url"]:
            ok = download_pdf(session, detail["pdf_url"], pdf_path)
            detail["pdf_saved"] = ok
        else:
            detail["pdf_saved"] = False
            log.warning("No PDF URL for: %s", detail["title"][:60])

        results.append({
            "title": detail["title"],
            "doi": detail["doi"],
            "authors": detail["authors"],
            "abstract": detail["abstract"],
            "journal": detail["journal"],
            "volume": detail["volume"],
            "issue": detail["issue"],
            "pub_date": detail["pub_date"],
            "articleno": detail["articleno"],
            "bib_key": bib_key,
            "pdf_saved": detail["pdf_saved"],
        })

        time.sleep(DELAY)

    # Save JSON
    with open(output_dir / "articles.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("Saved %d articles to %s", len(results), output_dir / "articles.json")

    pdf_count = sum(1 for r in results if r["pdf_saved"])
    log.info("Done! %d/%d PDFs downloaded.", pdf_count, len(results))


def main():
    parser = argparse.ArgumentParser(description="Springer Journal Article Crawler")
    parser.add_argument(
        "--journal",
        default="778",
        help="Springer journal ID (default: 778 = The VLDB Journal). "
             "Find it from the URL: link.springer.com/journal/<ID>",
    )
    parser.add_argument(
        "--volume",
        default="34",
        help="Volume number to download (default: 34)",
    )
    parser.add_argument(
        "--cookies",
        choices=["chrome", "firefox", "edge", "opera", "none"],
        default="chrome",
        help="Load browser cookies to access paywalled PDFs (default: chrome). "
             "Use 'none' to skip cookies (Open Access only).",
    )
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="Only retry downloading PDFs that failed previously",
    )
    args = parser.parse_args()

    list_url = f"{BASE_URL}/journal/{args.journal}/articles?filter-by-volume={args.volume}"
    output_dir = get_output_dir(args.journal, args.volume)

    session = requests.Session()

    # Load cookies by default (unless --cookies none is specified)
    if args.cookies != "none":
        cookies = load_browser_cookies(args.cookies)
        if cookies:
            for name, value in cookies.items():
                session.cookies.set(name, value, domain=".springer.com")
            log.info("Session now has %d cookies", len(session.cookies))
        else:
            log.warning("No cookies loaded. PDFs behind paywall will not be downloadable.")

    if args.retry_missing:
        # For retry, use existing output dir
        existing = list(PAPERS_DIR.glob("*/articles.json"))
        if existing:
            output_dir = existing[0].parent
        retry_missing_pdfs(session, output_dir)
    else:
        full_crawl(session, list_url, output_dir)


if __name__ == "__main__":
    main()