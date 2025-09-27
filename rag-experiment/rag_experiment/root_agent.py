from dataclasses import dataclass

from pydantic import BaseModel

from rag_experiment.context_based_answering_agent import ContextBasedAnsweringAgent
from rag_experiment.query_context_provider import QueryContextProvider

class RootAgentResponse(BaseModel):
    answer: str
    found_answer: bool

@dataclass
class RootAgent:
    context_based_answering_agent: ContextBasedAnsweringAgent
    query_context_provider: QueryContextProvider

    def respond(self, query_text: str) -> RootAgentResponse:
        query_context = self.query_context_provider.get_context(query_text)
        response = self.context_based_answering_agent.respond(
            query=query_text,
            context=query_context)
        return RootAgentResponse(
            answer=response.answer,
            found_answer=response.found_answer)
