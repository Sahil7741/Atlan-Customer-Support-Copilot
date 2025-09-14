from transformers import pipeline
import torch


class TicketSummarizer:

	def __init__(self, model_name: str = "facebook/bart-large-cnn"):
		device = 0 if torch.cuda.is_available() else -1

		try:
			self.pipe = pipeline(
				"summarization",
				model=model_name,
				device=device,
				model_kwargs={"torch_dtype": torch.float16} if device == 0 else {}
			)
		except Exception as e:
			print(f"[Summarizer] Using fallback model")
			self.pipe = pipeline("summarization", model="t5-base", device=device)

	def summarize(self, text: str) -> str:
		text = text.strip()
		if not text or len(text.split()) < 10:
			return text

		try:
			res = self.pipe(
				text,
				max_length=120,
				min_length=30,
				do_sample=False,
				early_stopping=True,
				no_repeat_ngram_size=2
			)
			return res[0]["summary_text"].strip()
		except Exception as e:
			print(f"[Summarizer] Using fallback summarization")
			words = text.split()
			if len(words) > 20:
				return " ".join(words[:20]) + "..."
			return text


