from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
import yaml, os, re, time, requests
from functools import lru_cache
from bs4 import BeautifulSoup
from transformers import pipeline as hf_pipeline
from .knowledge_base_scraper import AtlanKnowledgeBaseScraper

try:
	from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except Exception:
	from langchain.embeddings.openai import OpenAIEmbeddings
	from langchain.chat_models import ChatOpenAI

try:
	from langchain_groq import ChatGroq
except Exception:
	ChatGroq = None

class RAGAgent:

	def __init__(self, embed_model: str, gen_model: str, k: int = 4):
		self.embed_model = embed_model
		self.gen_model = gen_model
		self.k = k
		self.knowledge_base = None
		self.knowledge_base_scraper = None

	def _initialize_knowledge_base(self, urls: List[str]) -> FAISS:
		"""Initialize or load the knowledge base"""
		cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "config.yaml"))
		try:
			with open(cfg_path) as f:
				cfg = yaml.safe_load(f)
		except Exception:
			cfg = {}

		rag_config = cfg.get("rag", {})
		use_kb = rag_config.get("use_knowledge_base", True)
		kb_path = rag_config.get("knowledge_base_path", "atlan_knowledge_base")

		if use_kb:
			if os.path.exists(kb_path):
				print(f"[RAG] Loading knowledge base...")
				try:
					if os.getenv("OPENAI_API_KEY"):
						from langchain_openai import OpenAIEmbeddings
						embeddings = OpenAIEmbeddings(model=self.embed_model)
					else:
						embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

					kb = FAISS.load_local(kb_path, embeddings, allow_dangerous_deserialization=True)
					print(f"[RAG] Knowledge base loaded successfully")
					return kb
				except Exception as e:
					print(f"[RAG] Failed to load knowledge base, creating new one...")

			print(f"[RAG] Creating new knowledge base...")
			if not self.knowledge_base_scraper:
				max_depth = rag_config.get("max_scrape_depth", 2)
				max_pages = rag_config.get("max_scrape_pages", 50)
				self.knowledge_base_scraper = AtlanKnowledgeBaseScraper(max_depth=max_depth, max_pages=max_pages)

			pages = self.knowledge_base_scraper.discover_and_scrape(urls)

			kb = self.knowledge_base_scraper.create_knowledge_base(self.embed_model)

			if kb:
				self.knowledge_base_scraper.save_knowledge_base(kb, kb_path)
				print(f"[RAG] Knowledge base created successfully")
				return kb

		print(f"[RAG] Using fallback scraping method")
		return self._build_vs(tuple(urls))

	def _scrape_with_requests(self, url: str) -> str:
		"""Scrape using requests with better headers and retry logic"""
		headers = {
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

		for attempt in range(3):
			try:
				response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
				response.raise_for_status()

				soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')

				for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
					script.decompose()

				text = soup.get_text()

				text = self._clean_text(text)

				if len(text.split()) > 50:
					return text
				else:
					print(f"[RAG] Retrying scraping...")
					time.sleep(2)

			except Exception as e:
				print(f"[RAG] Scraping attempt failed, retrying...")
				if attempt < 2:
					time.sleep(2)

		return ""

	def _scrape_with_selenium(self, url: str) -> str:
		"""Scrape using Selenium for JavaScript-heavy sites"""
		try:
			from selenium import webdriver
			from selenium.webdriver.chrome.options import Options
			from selenium.webdriver.common.by import By
			from selenium.webdriver.support.ui import WebDriverWait
			from selenium.webdriver.support import expected_conditions as EC
			from webdriver_manager.chrome import ChromeDriverManager
			from selenium.webdriver.chrome.service import Service

			chrome_options = Options()
			chrome_options.add_argument("--headless")
			chrome_options.add_argument("--no-sandbox")
			chrome_options.add_argument("--disable-dev-shm-usage")
			chrome_options.add_argument("--disable-gpu")
			chrome_options.add_argument("--window-size=1920,1080")
			chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

			service = Service(ChromeDriverManager().install())
			driver = webdriver.Chrome(service=service, options=chrome_options)

			try:
				driver.get(url)

				WebDriverWait(driver, 10).until(
					EC.presence_of_element_located((By.TAG_NAME, "body"))
				)

				time.sleep(3)

				page_source = driver.page_source

				soup = BeautifulSoup(page_source, 'html.parser')

				for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
					script.decompose()

				text = soup.get_text()

				text = self._clean_text(text)

				return text

			finally:
				driver.quit()

		except Exception as e:
			print(f"[RAG] Selenium scraping failed, using fallback...")
			return ""

	def _clean_text(self, text: str) -> str:
		"""Clean and normalize text content"""
		import re
		import unicodedata

		text = unicodedata.normalize('NFKD', text)

		text = ''.join(char for char in text if unicodedata.category(char)[0] not in ['C'] or char in '\n\t ')

		text = re.sub(r'\s+', ' ', text)

		lines = text.split('\n')
		clean_lines = []
		for line in lines:
			line = line.strip()
			if len(line) < 5:
				continue
			alpha_chars = sum(1 for c in line if c.isalpha())
			if alpha_chars > len(line) * 0.1:
				clean_lines.append(line)

		text = ' '.join(clean_lines)

		text = re.sub(r'\s+', ' ', text).strip()

		return text

	@lru_cache(maxsize=10)
	def _build_vs(self, urls_key: tuple) -> FAISS:
		urls = list(urls_key)
		print(f"[RAG] Scraping {len(urls)} URLs...")

		docs = []
		successful_scrapes = 0

		for url in urls:

			content = ""

			content = self._scrape_with_requests(url)
			if content and len(content.split()) > 50:
				docs.append(Document(page_content=content, metadata={"source": url}))
				successful_scrapes += 1
				continue

			content = self._scrape_with_selenium(url)
			if content and len(content.split()) > 50:
				docs.append(Document(page_content=content, metadata={"source": url}))
				successful_scrapes += 1
				continue
			try:
				headers = {
					"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
					"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
					"Accept-Language": "en-US,en;q=0.9",
					"Accept-Encoding": "gzip, deflate, br",
					"Connection": "keep-alive",
				}
				
				loader = WebBaseLoader(
					[url],
					requests_kwargs={"timeout": 30, "headers": headers},
					bs_get_text_kwargs={"separator": " ", "strip": True}
				)
				fallback_docs = loader.load()
				if fallback_docs and len(fallback_docs[0].page_content.split()) > 50:
					docs.append(fallback_docs[0])
					successful_scrapes += 1
					continue
			except Exception as e:
				pass

			docs.append(Document(
				page_content=f"Documentation for {url} - content unavailable due to scraping restrictions",
				metadata={"source": url}
			))
		
		print(f"[RAG] Scraped {successful_scrapes}/{len(urls)} URLs successfully")

		filtered_docs = []
		for doc in docs:
			content = doc.page_content

			nav_indicators = ["skip to main", "search", "partner with us", "what's new", "support portal", "cookie", "privacy policy", "terms of service", "contact support", "get started", "connect data", "use data", "build governance", "configure atlan", "discover, trust"]
			if any(indicator in content.lower() for indicator in nav_indicators):
				continue

			if len(content.split()) < 50:
				continue

			words = content.split()
			if len(words) > 20:
				unique_words = len(set(word.lower() for word in words))
				if unique_words < len(words) * 0.4:
					continue

			if len(words) > 15:
				caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
				if caps_words > len(words) * 0.2:
					continue

			useful_terms = ["how to", "step", "configure", "setup", "install", "guide", "tutorial", "example", "integration", "connect", "lineage", "airflow", "dag", "etl", "pipeline", "api", "sdk", "endpoint", "response", "request", "authentication", "programmatically", "snowflake", "permissions", "credentials", "connection", "atlan", "documentation", "help", "support"]
			if any(term in content.lower() for term in useful_terms):
				filtered_docs.append(doc)
			elif len(content.split()) > 100:
				filtered_docs.append(doc)
		
		if not filtered_docs:
			filtered_docs = docs
		
		if not filtered_docs or all(len(doc.page_content.split()) < 50 for doc in filtered_docs):
			print("[RAG Debug] Adding fallback Atlan knowledge due to poor scraping")
			fallback_content = """
			Atlan is a modern data catalog and governance platform. Key features include:
			- Data discovery and profiling
			- Automated lineage tracking
			- Data quality monitoring
			- Integration with popular tools like Airflow, dbt, Fivetran, Tableau
			- API and SDK for custom integrations
			- SSO and access control
			- Glossary and business term management
			- Sensitive data detection and classification
			"""
			filtered_docs.append(Document(
				page_content=fallback_content,
				metadata={"source": "atlan-knowledge-base"}
			))
		
		splitter = RecursiveCharacterTextSplitter(
			chunk_size=800,
			chunk_overlap=100,
			length_function=len,
			separators=["\n\n", "\n", ". ", " ", ""]
		)
		chunks = splitter.split_documents(filtered_docs)

		quality_chunks = []
		for chunk in chunks:
			content = chunk.page_content
			words = content.split()

			if len(words) < 20:
				continue

			if len(words) > 10:
				unique_words = len(set(word.lower() for word in words))
				if unique_words < len(words) * 0.5:
					continue

			if len(words) > 15:
				caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
				if caps_words > len(words) * 0.15:
					continue

			quality_chunks.append(chunk)

		chunks = quality_chunks
		try:
			return FAISS.from_documents(chunks, OpenAIEmbeddings(model=self.embed_model))
		except Exception as e:
			try:
				hf = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
				return FAISS.from_documents(chunks, hf)
			except Exception:
				raise RuntimeError(f"Embedding failed: {e}")

	def answer(self, question: str, urls: List[str], response_style: str = "auto", use_intents: bool = False, topic_hint: str = "", analysis: dict = None) -> Tuple[str, List[str]]:
		
		if use_intents:
			q_lower = question.lower()
			cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "config.yaml"))
			try:
				with open(cfg_path) as f:
					cfg = yaml.safe_load(f)
			except Exception:
				cfg = {}
			intents = cfg.get("intents", [])
			for intent in intents:
				patterns = [p.lower() for p in intent.get("patterns", [])]
				requires_all = [p.lower() for p in intent.get("requires_all", [])]
				if requires_all and not all(r in q_lower for r in requires_all):
					continue
				if any(p in q_lower for p in patterns):
					return intent.get("response", ""), list(dict.fromkeys(urls))[:4]

		vs = self._initialize_knowledge_base(urls)
		search_k = max(self.k, 20) if any(term in question.lower() for term in ["airflow", "dag", "etl", "integration", "api", "sdk", "lineage", "endpoint", "programmatically", "missing", "upstream", "downstream", "snowflake", "permissions", "crawler"]) else max(self.k, 10)

		all_docs = []

		try:
			retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": search_k, "fetch_k": search_k * 3})
			mmr_docs = retriever.invoke(question)
			all_docs.extend(mmr_docs)
		except:
			pass

		try:
			retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": search_k})
			sim_docs = retriever.invoke(question)
			all_docs.extend(sim_docs)
		except:
			pass

		seen_content = set()
		rel = []
		for doc in all_docs:
			content_hash = hash(doc.page_content)
			if content_hash not in seen_content:
				seen_content.add(content_hash)
				rel.append(doc)
				if len(rel) >= search_k:
					break
		sources = []
		context = []
		for d in rel:
			u = d.metadata.get("source") or d.metadata.get("url") or ""
			if u: sources.append(u)
			context.append(d.page_content)
		
		seen = set(); uniq = []
		for s in sources:
			if s not in seen:
				seen.add(s); uniq.append(s)
		q_terms = {w.strip('.,()').lower() for w in question.split() if len(w) > 3}
		tech_terms = {
			"airflow": ["dag", "etl", "pipeline", "workflow", "operator", "task"],
			"integration": ["connect", "api", "sdk", "provider", "connector"],
			"lineage": ["relationship", "dependency", "flow", "mapping"]
		}
		expanded_terms = set(q_terms)
		for term in list(q_terms):
			if term in tech_terms:
				expanded_terms.update(tech_terms[term])
		q_terms = expanded_terms
		
		def is_good_content(text):
			text_lower = text.lower()
			nav_terms = ["skip to main", "search", "partner with us", "what's new", "support portal", "contact support", "get started", "connect data", "use data", "build governance", "configure atlan", "discover, trust", "cookie", "privacy policy", "terms of service"]
			if any(term in text_lower for term in nav_terms):
				return False
			if len(text.split()) < 20:
				return False
			if text.count(" ") < 15 and any(term in text_lower for term in ["manage", "data mesh", "governance", "custom metadata"]):
				return False
			words = text.split()
			if len(words) > 15:
				caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
				if caps_words > len(words) * 0.3:
					return False
			api_terms = ["entity_create", "entity_update", "authpolicy", "businesspolicy", "specifications", "types types core core", "entity_", "auth_", "business_"]
			if any(term in text_lower for term in api_terms):
				return False

			if len(words) > 10:
				word_counts = {}
				for word in words:
					word_counts[word.lower()] = word_counts.get(word.lower(), 0) + 1
				repeated_words = sum(1 for count in word_counts.values() if count > 2)
				if repeated_words > len(words) * 0.1:
					return False

			if len(words) > 20:
				unique_words = len(set(word.lower() for word in words))
				if unique_words < len(words) * 0.3:
					return False

			user_terms = ["how to", "step", "configure", "setup", "install", "guide", "tutorial", "example", "integration", "connect", "lineage", "airflow", "dag", "etl", "pipeline", "api", "sdk", "endpoint", "response", "request", "authentication", "programmatically", "snowflake", "permissions", "credentials", "connection"]
			return any(term in text_lower for term in user_terms) or len(text.split()) > 40
		
		filtered_context = [c for c in context if is_good_content(c)]
		if not filtered_context:
			filtered_context = context
		
		scored = sorted(filtered_context, key=lambda t: sum(1 for w in q_terms if w in t.lower()), reverse=True)
		MAX_CHARS = 2000 if any(term in question.lower() for term in ["airflow", "dag", "etl"]) else 1500
		ctx_parts = []
		cur = 0
		for chunk in scored:
			if cur >= MAX_CHARS:
				break
			chunk_words = chunk.split()
			if len(chunk_words) < 10:
				continue
			unique_words = len(set(word.lower() for word in chunk_words))
			if unique_words < len(chunk_words) * 0.6:
				continue
			ctx_parts.append(chunk)
			cur += len(chunk)

		ctx = "\n\n---\n\n".join(ctx_parts[:6])

		print(f"[RAG] Processing context...")
		if len(ctx) < 100:
			print("[RAG] Using fallback context")
			ctx = "Atlan is a modern data catalog and governance platform. For Snowflake integration, ensure proper permissions and crawler configuration."


		readable_chars = sum(1 for c in ctx if c.isprintable() or c.isspace())
		readable_ratio = readable_chars / len(ctx) if len(ctx) > 0 else 0

		alpha_chars = sum(1 for c in ctx if c.isalpha())
		alpha_ratio = alpha_chars / len(ctx) if len(ctx) > 0 else 0

		question_terms = set(question.lower().split())
		context_terms = set(ctx.lower().split())

		common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their"}
		meaningful_question_terms = question_terms - common_words
		meaningful_context_terms = context_terms - common_words
		relevant_overlap = len(meaningful_question_terms.intersection(meaningful_context_terms))

		technical_keywords = {"api", "lineage", "endpoint", "programmatically", "extract", "metadata", "snowflake", "permissions", "crawler", "upstream", "downstream", "integration", "sdk", "authentication", "token", "response", "json", "structure"}
		question_technical_terms = meaningful_question_terms.intersection(technical_keywords)
		context_technical_terms = meaningful_context_terms.intersection(technical_keywords)
		technical_overlap = len(question_technical_terms.intersection(context_technical_terms))

		garbled_ratio = sum(1 for c in ctx if not c.isalnum() and not c.isspace() and c not in '.,!?;:()[]{}"\'') / len(ctx) if ctx else 0

		question_lower = question.lower()
		is_api_question = any(term in question_lower for term in ["api", "sdk", "rest", "curl", "python", "requests", "custom asset", "entity", "create"])

		if is_api_question:
			if readable_ratio >= 0.3 and alpha_ratio >= 0.03 and relevant_overlap >= 1 and garbled_ratio <= 0.6:
				print(f"[RAG Debug] Using scraped knowledge base content for API question (readable: {readable_ratio:.2f}, alpha: {alpha_ratio:.2f}, meaningful overlap: {relevant_overlap}, technical overlap: {technical_overlap})")
			else:
				curated_knowledge = self._get_curated_knowledge(question)
				print(f"[RAG Debug] Context appears garbled or irrelevant for API question (readable: {readable_ratio:.2f}, alpha: {alpha_ratio:.2f}, meaningful overlap: {relevant_overlap}, technical overlap: {technical_overlap}, garbled: {garbled_ratio:.2f}), using curated knowledge base")
				ctx = curated_knowledge
		else:
			if readable_ratio < 0.4 or alpha_ratio < 0.05 or relevant_overlap < 1 or (len(question_technical_terms) > 0 and technical_overlap == 0) or garbled_ratio > 0.5:
				print(f"[RAG Debug] Context appears garbled or irrelevant (readable: {readable_ratio:.2f}, alpha: {alpha_ratio:.2f}, meaningful overlap: {relevant_overlap}, technical overlap: {technical_overlap}, garbled: {garbled_ratio:.2f}), using curated knowledge base")
				curated_knowledge = self._get_curated_knowledge(question)
				if len(curated_knowledge.split()) > 50:
					print("[RAG Debug] Returning curated knowledge directly for technical query")
					return curated_knowledge, list(dict.fromkeys(urls))[:4]
				ctx = curated_knowledge
			else:
				print(f"[RAG Debug] Using scraped knowledge base content (readable: {readable_ratio:.2f}, alpha: {alpha_ratio:.2f}, meaningful overlap: {relevant_overlap}, technical overlap: {technical_overlap})")
		style_instr = {
			"auto": "",
			"formal": "Use a professional, concise tone suitable for email.",
			"casual": "Use a friendly, conversational tone suitable for chat."
		}.get(response_style, "")

		policy = ""
		if topic_hint in {"API/SDK", "SSO", "Product", "How-to", "Best practices"}:
			policy = (
				"Keep the answer under 120 words. Provide 3–6 short steps. "
				"Cite relevant endpoints or pages by name if present in context. "
			)
		elif any(term in question.lower() for term in ["airflow", "dag", "etl", "integration"]):
			policy = (
				"Focus on specific configuration steps and technical details. "
				"Provide actionable installation/configuration instructions. "
				"Keep under 150 words with clear numbered steps. "
			)

		prompt = ChatPromptTemplate.from_messages([
			("system", (
				f"You are Atlan's helpful support copilot. {style_instr} {policy}"
				"Answer the user's question using the provided context from Atlan documentation. "
				"Provide a clear, helpful response with specific steps or guidance. "
				"If the context doesn't contain enough information to answer the question, respond with: 'I don't have enough information in the docs provided.' "
				"Keep your answer concise (under 150 words) and focused on the user's specific question. "
				"Use the context to provide relevant information, but synthesize it into a coherent response. "
				"Give a direct, actionable answer that helps the user solve their specific problem. "
				"Use simple, clear language that a customer support agent would use. "
				"Focus on practical solutions and troubleshooting steps. "
				"Always provide a helpful response based on the context provided. "
				"Don't repeat content verbatim - create a helpful, synthesized response."
			)),
			("user", "Question: {q}\n\nContext from Atlan docs:\n{c}\n\nAnswer:")
		])

		try:
			if ChatGroq is not None and os.getenv("GROQ_API_KEY"):
				groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
				llm = ChatGroq(model=groq_model, temperature=0)
			else:
				llm = ChatOpenAI(model=self.gen_model, temperature=0)
			resp = llm.invoke(prompt.format_messages(q=question, c=ctx), config={"timeout": 20})
			raw_answer = resp.content.strip()
			ans = self._sanitize(raw_answer, ctx, question, analysis)
			
			if ans == "I don't have enough information in the docs provided.":
				retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 12, "fetch_k": 20})
				rel = retriever.invoke(question)
				sources = []
				context = []
				for d in rel:
					u = d.metadata.get("source") or d.metadata.get("url") or ""
					if u: sources.append(u)
					context.append(d.page_content)
				
				ctx_parts = []
				cur = 0
				for chunk in context[:15]:
					if cur >= 4000:
						break
					ctx_parts.append(chunk)
					cur += len(chunk)
				ctx = "\n\n---\n\n".join(ctx_parts)
				
				resp = llm.invoke(prompt.format_messages(q=question, c=ctx), config={"timeout": 20})
				ans = self._sanitize(resp.content.strip(), ctx, question, analysis)
			
			return ans, uniq[:6]
		except Exception as e:
			try:
				gen = hf_pipeline("text2text-generation", model="google/flan-t5-base")
				local_prompt = (
					"Answer the user based ONLY on the provided context. "
					"If the answer isn't present, say you don't have enough info. Keep it under 120 words.\n\n"
					f"Question:\n{question}\n\nContext (trimmed):\n{ctx[:2000]}\n\nAnswer succinctly:"
				)
				out = gen(local_prompt, max_length=256, do_sample=False)
				text = out[0]["generated_text"].strip()
				ans = self._sanitize(text, ctx, question, analysis)
				return ans, uniq[:6]
			except Exception:
				raise RuntimeError(f"Generation failed: {e}")

	def _sanitize(self, answer_text: str, full_context: str, original_question: str = "", analysis: dict = None) -> str:
		"""Sanitize RAG response and filter out garbled content."""
		blacklist = ["cognite", "snowplow", "amundsen", "datahub", "collibra"]
		lines = [ln for ln in answer_text.splitlines() if not any(b in ln.lower() for b in blacklist)]
		clean = "\n".join([ln for ln in lines if ln.strip()])
		
		words = clean.split()
		if len(words) > 8:
			caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
			if caps_words > len(words) * 0.4:
				print("[RAG Debug] Detected garbled API documentation pattern, rejecting")
				return "I don't have enough information in the docs provided."
		
		garbled_patterns = [
			"ENTITY_CREATE ENTITY_UPDATE",
			"AuthPolicy AuthService BusinessPolicy",
			"Specifications Specifications",
			"Types Types Core Core",
			"entity_create entity_update",
			"authpolicy authservice businesspolicy"
		]
		for pattern in garbled_patterns:
			if pattern in clean:
				print(f"[RAG Debug] Detected garbled pattern '{pattern}', rejecting")
				return "I don't have enough information in the docs provided."
		
		if len(words) > 8:
			word_counts = {}
			for word in words:
				word_counts[word.lower()] = word_counts.get(word.lower(), 0) + 1
			repeated_words = sum(1 for count in word_counts.values() if count > 2)
			if repeated_words > len(words) * 0.25:
				print("[RAG Debug] Detected repeated words pattern, rejecting")
				return "I don't have enough information in the docs provided."
		
		tech_terms = ["entity", "auth", "business", "policy", "specification", "type", "core"]
		if len(words) > 10:
			tech_count = sum(1 for word in words if any(term in word.lower() for term in tech_terms))
			if tech_count > len(words) * 0.5:
				print("[RAG Debug] Detected excessive technical terms without context, rejecting")
				return "I don't have enough information in the docs provided."

		if "```" in clean or "`" in clean:
			if len(clean.split()) < 3:
				return "I don't have enough information in the docs provided."
			return clean

		if len(clean.split()) < 5:
			return "I don't have enough information in the docs provided."
		
		question_words = set(original_question.lower().split())
		answer_words = set(clean.lower().split())
		overlap = len(question_words.intersection(answer_words))

		technical_terms = ["lineage", "upstream", "downstream", "missing", "crawler", "permissions", "snowflake", "view", "table", "atlan", "integration", "api", "sdk", "help", "problem", "issue", "troubleshoot"]
		has_technical_terms = any(term in question_words for term in technical_terms)

		if len(clean.split()) > 10:
			char_counts = {}
			for char in clean:
				char_counts[char] = char_counts.get(char, 0) + 1
			max_char_count = max(char_counts.values()) if char_counts else 0
			if max_char_count > len(clean) * 0.3:
				print("[RAG Debug] Response appears garbled, using curated knowledge")
				return self._get_curated_knowledge(original_question)

		alpha_chars = sum(1 for c in clean if c.isalpha())
		if len(clean) > 20 and alpha_chars < len(clean) * 0.3:
			print("[RAG Debug] Response has too few alphabetic characters, using curated knowledge")
			return self._get_curated_knowledge(original_question)

		printable_chars = sum(1 for c in clean if c.isprintable() or c.isspace())
		if len(clean) > 20 and printable_chars < len(clean) * 0.7:
			print("[RAG Debug] Response has too many non-printable characters, using curated knowledge")
			return self._get_curated_knowledge(original_question)

		if has_technical_terms:
			if len(clean.split()) < 3:
				print("[RAG Debug] Technical query response too short, using curated knowledge")
				return self._get_curated_knowledge(original_question)
			return clean
		else:
			if overlap == 0 and len(question_words) > 8 and len(clean.split()) < 5:
				print("[RAG Debug] Response doesn't seem to address the question and is too short, using curated knowledge")
				return self._get_curated_knowledge(original_question)

		return clean

	def _get_curated_knowledge(self, question: str) -> str:
		"""Get curated knowledge based on question keywords"""
		question_lower = question.lower()

		if any(term in question_lower for term in ["snowflake", "connect", "integration"]):
			return """
			Atlan Snowflake Integration Guide:

			Prerequisites:
			- Snowflake account with admin access
			- Network connectivity between Atlan and Snowflake

			Setup Steps:
			1. Create a dedicated user in Snowflake (e.g., ATLAN_USER)
			2. Create a role (e.g., ATLAN_READER) with appropriate permissions
			3. Grant USAGE on databases and schemas you want to crawl
			4. Grant SELECT on tables and views
			5. Ensure the role has access to the warehouse

			Connection Configuration:
			- Account identifier (e.g., abc12345.us-east-1)
			- Username and password
			- Warehouse name
			- Database and schema to crawl

			Common Issues:
			- Permission errors: Ensure role has SELECT on all objects
			- Network issues: Check firewall and VPN settings
			- Warehouse access: Ensure role can use the warehouse
			"""

		elif any(term in question_lower for term in ["api", "lineage", "programmatically", "endpoint", "extract", "metadata", "custom asset", "entity", "create", "rest", "curl", "python", "requests"]):
			return """
			Atlan API and Custom Asset Creation:

			Creating Custom Assets via REST API:

			For creating custom assets like 'Report' entities, you'll need to use the Atlan REST API with proper authentication and payload structure.

			Basic cURL Example:
			```bash
			curl -X POST "https://your-tenant.atlan.com/api/meta/entity" \
			  -H "Authorization: Bearer YOUR_API_TOKEN" \
			  -H "Content-Type: application/json" \
			  -d '{
			    "typeName": "Report",
			    "attributes": {
				  "name": "My Custom Report",
				  "qualifiedName": "report/my-custom-report",
				  "description": "A custom report asset"
				}
			  }'
			```

			Python requests Example:
			```python
			import requests

			url = "https://your-tenant.atlan.com/api/meta/entity"
			headers = {
				"Authorization": "Bearer YOUR_API_TOKEN",
				"Content-Type": "application/json"
			}
			payload = {
				"typeName": "Report",
				"attributes": {
					"name": "My Custom Report",
					"qualifiedName": "report/my-custom-report",
					"description": "A custom report asset"
				}
			}

			response = requests.post(url, json=payload, headers=headers)
			print(response.json())
			```

			Key Requirements:
			- Valid API token with appropriate permissions
			- Correct typeName for your custom asset type
			- Unique qualifiedName following Atlan naming conventions
			- Proper JSON structure in request body

			Common Issues:
			- 400 errors often indicate missing required fields or invalid payload structure
			- Ensure your API token has entity creation permissions
			- Check that the custom asset type is properly configured in Atlan

			For detailed API documentation and SDK examples, refer to the Atlan Developer Hub at developer.atlan.com.
			"""

		elif any(term in question_lower for term in ["upstream", "downstream", "missing", "crawler"]):
			return """
			Atlan Lineage Troubleshooting Guide:

			Common Causes of Missing Lineage:
			1. Permission Issues:
			   - Crawler lacks SELECT permissions on upstream tables
			   - Schema access not granted for related objects
			   - Cross-database lineage requires additional permissions

			2. Crawler Configuration:
			   - Crawler not configured to detect relationships
			   - Lineage detection disabled in crawler settings
			   - Crawler running on wrong schedule

			3. Object Issues:
			   - Upstream tables don't exist in expected schemas
			   - Views referencing non-existent tables
			   - Temporary tables or views not persisted

			4. Snowflake-Specific Issues:
			   - Views created with DEFINER rights
			   - Cross-account lineage not supported
			   - External tables not accessible

			Troubleshooting Steps:
			1. Verify crawler permissions on all related objects
			2. Check crawler configuration for lineage detection
			3. Re-run crawler after permission changes
			4. Verify all referenced objects exist
			5. Check Snowflake query history for view definitions
			"""

		else:
			return """
			Atlan Data Catalog and Governance Platform:

			Atlan is a modern data catalog and governance platform that helps organizations discover, understand, and trust their data. It provides automated data discovery, lineage tracking, and governance capabilities.

			Key Features:
			- Automated data discovery and cataloging
			- Data lineage tracking and visualization
			- Data quality monitoring and governance
			- Collaborative data documentation
			- Integration with 100+ data sources including Snowflake, BigQuery, Redshift, and more

			Common Use Cases:
			- Data discovery and exploration
			- Impact analysis and lineage tracking
			- Data quality monitoring
			- Compliance and governance
			- Self-service analytics

			Getting Started:
			1. Connect your data sources
			2. Configure crawlers for automated discovery
			3. Set up data quality rules
			4. Create custom metadata and tags
			5. Enable collaboration features

			For technical questions about Atlan, you can refer to the official documentation or contact support for assistance.
			"""



