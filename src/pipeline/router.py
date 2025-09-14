from typing import Dict


def route_ticket(analysis: Dict) -> str:
	"""Return a routing message for non-RAG topics."""
	topic = next(iter(analysis.get("topic_tags", [])), "General")
	return f"This ticket has been classified as a '{topic}' issue and routed to the appropriate team."
	