from typing import List
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    encoding = tiktoken.get_encoding("cl100k_base")

    def tiktoken_len(text: str) -> int:
        return len(encoding.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]
