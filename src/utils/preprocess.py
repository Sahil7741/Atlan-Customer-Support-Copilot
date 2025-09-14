import io
import json
import pandas as pd


def load_tickets(file, filename: str | None = None) -> pd.DataFrame:
	"""Load tickets from CSV or JSON and normalize to columns: id, subject, body, text.

	Accepted schemas:
	- CSV with columns: id,subject,body
	- JSON array of objects with keys: id,subject,body
	- Backward-compat: CSV with id,channel,text (will map text->body and set subject="")
	"""
	name = (filename or getattr(file, "name", "")).lower()
	if name.endswith(".json"):
		data = json.load(file)
		df = pd.DataFrame(data)
	elif name.endswith(".csv") or not name:
		df = pd.read_csv(file)
	else:
		raise ValueError("Unsupported file type. Upload .csv or .json")

	cols_lower = {c.lower(): c for c in df.columns}
	# Normalize columns
	if {"id","subject","body"}.issubset(set(cols_lower.keys())):
		id_col = cols_lower["id"]; subj_col = cols_lower["subject"]; body_col = cols_lower["body"]
		df = df[[id_col, subj_col, body_col]].rename(columns={id_col:"id", subj_col:"subject", body_col:"body"})
	elif {"id","channel","text"}.issubset(set(cols_lower.keys())):
		# backward compat: no subject; use empty subject and map text->body
		id_col = cols_lower["id"]; text_col = cols_lower["text"]
		df = df[[id_col, text_col]].rename(columns={id_col:"id", text_col:"body"})
		df["subject"] = ""
	else:
		raise ValueError("CSV/JSON must contain columns id,subject,body (or legacy id,channel,text)")

	# Create unified text field for classification
	df["subject"] = df["subject"].fillna("").astype(str)
	df["body"] = df["body"].fillna("").astype(str)
	df["text"] = (df["subject"].str.strip() + ". " + df["body"].str.strip()).str.strip()
	return df


