import requests
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Set, Dict, Tuple
import json
from dataclasses import dataclass
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    from langchain.embeddings.openai import OpenAIEmbeddings


@dataclass
class ScrapedPage:
    url: str
    title: str
    content: str
    links: List[str]
    depth: int


class AtlanKnowledgeBaseScraper:

    def __init__(self, max_depth: int = 3, max_pages: int = 100):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.scraped_pages: List[ScrapedPage] = []
        self.base_domains = {
            "docs.atlan.com",
            "developer.atlan.com"
        }

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache"
        }

        self.priority_keywords = {
            "lineage", "api", "sdk", "integration", "airflow", "dbt", "fivetran",
            "tableau", "snowflake", "permissions", "authentication", "setup",
            "configuration", "troubleshooting", "guide", "tutorial", "getting-started",
            "connector", "crawler", "metadata", "governance", "quality", "glossary"
        }

        self.avoid_keywords = {
            "privacy", "terms", "cookie", "contact", "support", "login", "signup",
            "pricing", "about", "careers", "blog", "news", "press", "legal"
        }

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for scraping"""
        try:
            parsed = urlparse(url)
            if not parsed.netloc or not parsed.scheme:
                return False

            if parsed.netloc not in self.base_domains:
                return False

            if any(url.lower().endswith(ext) for ext in ['.pdf', '.zip', '.exe', '.dmg', '.png', '.jpg', '.gif']):
                return False

            avoid_paths = ['/api/', '/admin/', '/login/', '/signup/', '/pricing/', '/contact/']
            if any(path in url.lower() for path in avoid_paths):
                return False

            return True
        except:
            return False

    def should_prioritize_url(self, url: str, title: str = "", content: str = "") -> int:
        """Return priority score for URL (higher = more important)"""
        score = 0
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content.lower()

        if "docs.atlan.com" in url:
            score += 50
        elif "developer.atlan.com" in url:
            score += 5

        if any(keyword in url_lower for keyword in ["/lineage/", "/api/", "/sdk/", "/integration/", "/getting-started/", "/capabilities/", "/product/", "/governance/", "/security/"]):
            score += 15

        if any(keyword in url_lower for keyword in ["/airflow/", "/dbt/", "/fivetran/", "/tableau/", "/snowflake/", "/connector/", "/troubleshooting/", "/faq/"]):
            score += 10

        for keyword in self.priority_keywords:
            if keyword in title_lower:
                score += 3
            if keyword in content_lower:
                score += 1

        for keyword in self.avoid_keywords:
            if keyword in url_lower or keyword in title_lower:
                score -= 5

        if "developer.atlan.com" in url:
            score -= 30

        return score

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""

        text = re.sub(r'\s+', ' ', text)

        lines = text.split('\n')
        clean_lines = []

        for line in lines:
            line = line.strip()
            if len(line) < 5:
                continue

            alpha_chars = sum(1 for c in line if c.isalpha())
            total_chars = len(line)

            if total_chars > 0 and alpha_chars / total_chars > 0.2:
                clean_lines.append(line)

        if not clean_lines:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 10:
                    clean_lines.append(line)

        if not clean_lines:
            return ""

        text = ' '.join(clean_lines)

        text = re.sub(r'\s+', ' ', text).strip()

        text = re.sub(r'\b(Home|Back|Next|Previous|Menu|Search|Login|Sign in|Sign up)\b', '', text, flags=re.IGNORECASE)

        text = re.sub(r'\b\d+\s*px\b', '', text)
        text = re.sub(r'\b\d+\s*rem\b', '', text)
        text = re.sub(r'\b\d+\s*em\b', '', text)

        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def scrape_page(self, url: str, depth: int = 0) -> ScrapedPage:
        """Scrape a single page"""
        try:

            response = requests.get(url, headers=self.headers, timeout=30, allow_redirects=True)
            response.raise_for_status()

            response.encoding = response.apparent_encoding or 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')

            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()

            content = ""

            main_content = None

            docusaurus_selectors = [
                'div[class*="theme-doc-markdown"]',
                'div[class*="markdown"]',
                'div[class*="content"]',
                'div[class*="main"]',
                'article',
                'main',
                'div[class*="prose"]',
                'div[class*="documentation"]'
            ]

            for selector in docusaurus_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break

            if not main_content:
                content_divs = soup.find_all('div')
                for div in content_divs:
                    text_content = div.get_text().strip()
                    if len(text_content) > 200 and len(text_content.split()) > 30:
                        if not any(nav_word in text_content.lower() for nav_word in ['navigation', 'menu', 'sidebar', 'toc']):
                            main_content = div
                            break

            if main_content:
                for element in main_content(['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript', 'button']):
                    element.decompose()

                content = main_content.get_text()
            else:
                for element in soup(['nav', 'header', 'footer', 'aside', 'script', 'style', 'button']):
                    element.decompose()
                content = soup.get_text()

            content = self.clean_text(content)

            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                if self.is_valid_url(full_url):
                    links.append(full_url)

            return ScrapedPage(
                url=url,
                title=title,
                content=content,
                links=links,
                depth=depth
            )

        except Exception as e:
            print(f"[Scraper] Error scraping {url}: {e}")
            return ScrapedPage(url=url, title="", content="", links=[], depth=depth)

    def discover_and_scrape(self, start_urls: List[str]) -> List[ScrapedPage]:
        """Discover and scrape all relevant pages starting from seed URLs"""
        print(f"[Scraper] Starting web scraping...")

        url_queue = []
        for url in start_urls:
            if self.is_valid_url(url):
                url_queue.append((url, 0, 0))

        url_queue.sort(key=lambda x: x[2], reverse=True)

        while url_queue and len(self.scraped_pages) < self.max_pages:
            url, depth, priority = url_queue.pop(0)

            if url in self.visited_urls or depth > self.max_depth:
                continue

            self.visited_urls.add(url)

            page = self.scrape_page(url, depth)

            if len(page.content.split()) > 20:
                self.scraped_pages.append(page)

                if depth < self.max_depth:
                    for link in page.links:
                        if link not in self.visited_urls:
                            new_priority = self.should_prioritize_url(link, page.title, page.content)
                            url_queue.append((link, depth + 1, new_priority))

                    url_queue.sort(key=lambda x: x[2], reverse=True)

            time.sleep(1)

        print(f"[Scraper] Scraping complete: {len(self.scraped_pages)} pages")
        return self.scraped_pages

    def create_knowledge_base(self, embed_model: str = "text-embedding-3-small") -> FAISS:
        """Create a FAISS vector store from scraped pages"""
        print(f"[Scraper] Creating knowledge base...")

        documents = []
        for page in self.scraped_pages:
            if len(page.content.split()) > 20:
                doc = Document(
                    page_content=page.content,
                    metadata={
                        "source": page.url,
                        "title": page.title,
                        "depth": page.depth
                    }
                )
                documents.append(doc)

        if not documents:
            print("[Scraper] No documents to create knowledge base")
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_documents(documents)
        print(f"[Scraper] Created {len(chunks)} content chunks")

        try:
            if os.getenv("OPENAI_API_KEY"):
                embeddings = OpenAIEmbeddings(model=embed_model)
            else:
                embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

            vector_store = FAISS.from_documents(chunks, embeddings)
            print(f"[Scraper] Vector store created successfully")
            return vector_store

        except Exception as e:
            print(f"[Scraper] Error creating vector store: {e}")
            return None

    def save_knowledge_base(self, vector_store: FAISS, output_path: str = "atlan_knowledge_base"):
        """Save the knowledge base to disk"""
        try:
            vector_store.save_local(output_path)
            print(f"[Scraper] Knowledge base saved to {output_path}")

            metadata = {
                "scraped_pages": len(self.scraped_pages),
                "total_chunks": len(self.scraped_pages),
                "scraped_urls": [page.url for page in self.scraped_pages],
                "base_domains": list(self.base_domains)
            }

            with open(f"{output_path}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"[Scraper] Error saving knowledge base: {e}")

    def load_knowledge_base(self, input_path: str = "atlan_knowledge_base") -> FAISS:
        """Load a previously saved knowledge base"""
        try:
            vector_store = FAISS.load_local(input_path, HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"))
            print(f"[Scraper] Loaded knowledge base from {input_path}")
            return vector_store
        except Exception as e:
            print(f"[Scraper] Error loading knowledge base: {e}")
            return None


def main():
    """Main function to run the scraper"""
    seed_urls = [
        "https://docs.atlan.com/",
        "https://developer.atlan.com/",
        "https://docs.atlan.com/get-started/what-is-atlan",
        "https://docs.atlan.com/get-started/how-tos/quick-start-for-admins",
        "https://docs.atlan.com/apps/connectors/data-warehouses/snowflake/how-tos/set-up-snowflake",
        "https://docs.atlan.com/apps/connectors/data-warehouses/databricks/how-tos/set-up-databricks",
        "https://docs.atlan.com/apps/connectors/business-intelligence/microsoft-power-bi/how-tos/set-up-microsoft-power-bi",
        "https://docs.atlan.com/platform/references/atlan-architecture",
        "https://docs.atlan.com/product/integrations/automation/browser-extension/how-tos/use-the-atlan-browser-extension",
        "https://docs.atlan.com/secure-agent",
        "https://docs.atlan.com/product/capabilities/playbooks",
        "https://docs.atlan.com/product/capabilities/discovery",
        "https://docs.atlan.com/product/capabilities/governance/contracts",
        "https://docs.atlan.com/product/integrations",
        "https://docs.atlan.com/support/references/customer-support",
        # "https://docs.atlan.com/get-started/",
        # "https://docs.atlan.com/product/",
        # "https://docs.atlan.com/apps/",
        # "https://docs.atlan.com/platform/",
        # "https://docs.atlan.com/support/",
        "https://developer.atlan.com/getting-started/",
        "https://developer.atlan.com/sdks/",
    ]

    scraper = AtlanKnowledgeBaseScraper(max_depth=2, max_pages=200)

    pages = scraper.discover_and_scrape(seed_urls)

    vector_store = scraper.create_knowledge_base()

    if vector_store:
        scraper.save_knowledge_base(vector_store)
        print(f"[Scraper] Knowledge base creation complete")
    else:
        print(f"[Scraper] Failed to create knowledge base")


if __name__ == "__main__":
    main()
