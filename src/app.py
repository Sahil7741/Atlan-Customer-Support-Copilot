import os
import sys
try:
	from dotenv import load_dotenv
except Exception:
	def load_dotenv(*args, **kwargs):
		return False
import pandas as pd
import streamlit as st
import yaml
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.classify_infer import CustomerSupportClassifier
from pipeline.summarizer import TicketSummarizer
from pipeline.rag_agent import RAGAgent
from pipeline.router import route_ticket
from utils.preprocess import load_tickets


st.set_page_config(page_title="Atlan Support Copilot", page_icon="🛟", layout="wide")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=dotenv_path, override=False)

def _mask(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 8:
        return "*" * (len(k) - 2) + k[-2:]
    return k[:4] + "*" * (len(k) - 8) + k[-4:]

_oa = os.getenv("OPENAI_API_KEY") or ""
_gq = os.getenv("GROQ_API_KEY") or ""
_ua = os.getenv("USER_AGENT") or ""
print(f"[System] Initializing Atlan Customer Support Copilot...")
print(f"[System] API Configuration: OpenAI={bool(_oa)}, Groq={bool(_gq)}")


@st.cache_resource
def load_config():
	root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	cfg_path = os.path.join(root, "config.yaml")
	with open(cfg_path) as f:
		return yaml.safe_load(f)


cfg = load_config()

st.title("🛟 Atlan Customer Support Copilot")

with st.expander("⚙️ Configuration", expanded=False):
	col1, col2 = st.columns(2)
	with col1:
		st.caption("Classifier is local; RAG uses OpenAI or Groq.")
		use_intents = st.checkbox("Use intents (hardcoded responses)", value=False, help="If on, uses hardcoded responses from config.yaml. If off, uses true RAG with dynamic generation.")
	with col2:
		prov_openai = bool(os.getenv("OPENAI_API_KEY"))
		prov_groq = bool(os.getenv("GROQ_API_KEY"))
		ua = os.getenv("USER_AGENT") or ""
		st.markdown(f"**Status:**")
		st.markdown(f"- OpenAI configured: {'✅' if prov_openai else '❌'}")
		st.markdown(f"- Groq configured: {'✅' if prov_groq else '❌'}")
		st.markdown(f"- USER_AGENT set: {'✅' if ua else '❌'}")

@st.cache_resource
def init_components():
	classifier = CustomerSupportClassifier()
	summarizer = TicketSummarizer(model_name=cfg["summarization"]["model_name"])
	rag = RAGAgent(embed_model=cfg["rag"]["embed_model"], gen_model=cfg["rag"]["gen_model"], k=cfg["rag"]["k"])
	return classifier, summarizer, rag


classifier, summarizer, rag = init_components()

tab1, tab2 = st.tabs(["📊 Bulk Dashboard", "🤖 Interactive Agent"])

with tab1:
	st.subheader("Bulk Ticket Classification (Can take up to 5 minute to classify tickets)")

	@st.cache_data
	def load_and_classify_sample_tickets():
		default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data", "sample_tickets.csv"))
		if os.path.exists(default_path):
			with open(default_path, "r") as f:
				df = load_tickets(f, filename="sample_tickets.csv")
			preds = classifier.classify_many(df["text"].tolist())
			df_out = pd.concat([df, pd.DataFrame(preds)], axis=1)
			return df_out
		return None
	df_out = load_and_classify_sample_tickets()

	if df_out is not None:
		display_df = df_out.drop(columns=['text'], errors='ignore')
		st.dataframe(
			display_df,
			use_container_width=False,
			hide_index=True,
			height=len(display_df) * 35 + 40,
			column_config={
				"id": st.column_config.TextColumn("ID", width="medium"),
				"subject": st.column_config.TextColumn("Subject", width="medium"),
				"body": st.column_config.TextColumn("Body", width="medium"),
				"topic_tags": st.column_config.TextColumn("Topics", width="medium"),
				"sentiment": st.column_config.TextColumn("Sentiment", width="medium"),
				"priority": st.column_config.TextColumn("Priority", width="medium")
			}
		)
		st.download_button("Download results", df_out.to_csv(index=False), file_name="classified_tickets.csv")
		st.caption("✅ Classified 30 sample tickets successfully!")
	else:
		st.error("Sample tickets file not found")

	st.markdown("**Or upload your own tickets:**")
	upload = st.file_uploader("Upload tickets (.csv or .json with id,subject,body)", type=["csv","json"])
	if upload:
		with st.spinner("Processing uploaded tickets..."):
			df = load_tickets(upload, filename=upload.name)
			preds = classifier.classify_many(df["text"].tolist())
			df_out = pd.concat([df, pd.DataFrame(preds)], axis=1)
		st.dataframe(df_out, use_container_width=True, hide_index=True)
		st.download_button("Download results", df_out.to_csv(index=False), file_name="classified_tickets.csv")

	if not upload:
		st.info("👆 Upload your own CSV/JSON file to classify additional tickets")

with tab2:
	st.subheader("Interactive Agent (Can take up to 1 minute to respond to a ticket)")
	col1, col2 = st.columns([2,3])
	with col1:
		if "agent_channel" not in st.session_state:
			st.session_state.agent_channel = "email"
		if "agent_text" not in st.session_state:
			st.session_state.agent_text = ""
		if "agent_subject" not in st.session_state:
			st.session_state.agent_subject = ""

		with st.form("agent_form", clear_on_submit=False):
			channel = st.selectbox("Channel", ["email","chat","whatsapp","voice"], key="agent_channel")
			subject = st.text_input("Subject", key="agent_subject", placeholder="Enter ticket subject...")
			text = st.text_area("Ticket text", height=160, key="agent_text", placeholder="Enter your support ticket text here...")
			summarize_chk = st.checkbox("Add ticket summary", key="agent_summarize", value=True, disabled=True)
			submitted = st.form_submit_button("Analyze & Respond")
	with col2:
		if submitted and st.session_state.agent_text.strip():
			full_text = f"Subject: {st.session_state.agent_subject}\n\n{st.session_state.agent_text}" if st.session_state.agent_subject.strip() else st.session_state.agent_text
			analysis = classifier.classify_one(st.session_state.agent_text)
			if st.session_state.get("agent_summarize"):
				analysis["summary"] = summarizer.summarize(full_text)

			st.markdown("### Internal Analysis")
			st.json(analysis)
			rag_triggering_topics = {"How-to","Product","Best practices","API/SDK","SSO"}
			primary_topic = next(iter(analysis["topic_tags"]), "")
			if primary_topic not in rag_triggering_topics:
				msg = route_ticket(analysis)
				st.markdown("### Team Routing")
				st.info(msg)

			if os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"):
				try:
					answer, sources = rag.answer(
						full_text,
						cfg["rag"]["default_urls"],
						response_style="auto",
						use_intents=use_intents,
						topic_hint=primary_topic,
						analysis=analysis
					)
					st.markdown("### Final Response")
					st.write(answer)
					if sources:
						st.markdown("**Sources:**")
						for s in sources:
							st.write(f"- {s}")
				except Exception as e:
					st.error(str(e))
					st.info("Tip: If you see an OpenAI quota/rate limit error, update billing or use a different API key and retry.")
			else:
				st.warning("Set OPENAI_API_KEY or GROQ_API_KEY for RAG answers.")

# Footer
st.markdown("---")
st.markdown(
	'<div style="text-align: center; color: #FFFFFF; font-size: 0.9em; margin: 2rem auto; width: 100%; display: block;">'
	'Built with ❤️ for Atlan\'s Customer Support Automation<br>by <strong>Sahil Jaiswal</strong>'
	'</div>',
	unsafe_allow_html=True
)

