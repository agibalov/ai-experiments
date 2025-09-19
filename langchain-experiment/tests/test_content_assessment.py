import re
import tiktoken
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from trafilatura import fetch_url, extract
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate

from expert import check_text_entail_claim

def semantic_chunk_text(text: str, max_chunk_tokens: int, overlap_tokens: int = 64) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode

    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text).strip()

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    mk_sem = lambda: SemanticChunker(
        emb,
        breakpoint_threshold_type="gradient",
        breakpoint_threshold_amount=85,
        buffer_size=0,
    )

    sem_chunks: List[str] = []
    for p in paras:
        sem_chunks.extend(mk_sem().split_text(p))

    final: List[str] = []
    step = max(1, max_chunk_tokens - max(0, overlap_tokens))

    for ch in sem_chunks:
        ch_ids = toks(ch)
        if len(ch_ids) <= max_chunk_tokens:
            final.append(ch)
            continue

        for i in range(0, len(ch_ids), step):
            piece_ids = ch_ids[i:i + max_chunk_tokens]
            if not piece_ids:
                continue
            final.append(enc.decode(piece_ids))

    return final


def summarize_chunks_map_reduce(chunks: List[str]) -> str:
    MAP_PROMPT = PromptTemplate.from_template(
        "Summarize the key points of this passage in 1-5 concise bullets:\n\n{chunk}"
    )
    REDUCE_PROMPT = PromptTemplate.from_template("""
Here is a list of key ideas extracted from an article. Combine them into a concise 1 paragraph summary.
Make it clear that this is a summary of an article: use language like "this article mentions ...", 
"the author suggests ...", etc.

The list of key ideas:
{bullets}
""")
    llm = init_chat_model("ollama:gpt-oss:20b", temperature=0, seed=31337, num_ctx=4096)
    bullets = [ (MAP_PROMPT | llm).invoke({"chunk": c}).content for c in chunks ]
    
    for i in range(len(chunks)):
        print(f"""
***
CHUNK:
{chunks[i]}

BULLETS:
{bullets[i]}


              """)
    
    final = (REDUCE_PROMPT | llm).invoke({"bullets": "\n\n".join(bullets)}).content
    return final


ASSESS_WORTHINESS_PROMPT = PromptTemplate.from_template("""
You are a content recommending expert that determines if content
is worth reading based on reader's values and the content summary.

Respond with one sentence saying whether this content is worth reading, 
and why (or why not).

Reader's values:
{reader_values}

Content summary:
{summary}
""")

EXPLAIN_WORTHINESS_PROMPT = PromptTemplate.from_template("""
Interpret content with respect to reader's values. Emphasis 
on what's relevant to the reader. Exclude technical details and 
minor nuances.

Provide a concise answer - 2-3 sentences - as a plain text without formatting.

Reader's values:
{reader_values}

Content:
{summary}
""")

def test_content_assessment():
    clean_text = extract(fetch_url('https://agibalov.io/2017/05/26/Generate-Java-code-documentation-with-QDox-EJS-Nashorn-and-Asciidoctor/'))
    chunks = semantic_chunk_text(clean_text, 1000)
    summary = summarize_chunks_map_reduce(chunks)
    print(f"***SUMMARY\n{summary}***")
    
    llm = init_chat_model("ollama:gpt-oss:20b", temperature=0, seed=31337, num_ctx=4096)
    
    if True:
        reader_values = """
        * Beer
        * Python
        """
        
        result = (ASSESS_WORTHINESS_PROMPT | llm).invoke({
            "reader_values": reader_values, 
            "summary": summary
            })
        print("\n\nWORTH READING?")
        print(result.content)
        assert not check_text_entail_claim(result.content, "positive").is_true
        
        result = (EXPLAIN_WORTHINESS_PROMPT | llm).invoke({
            "reader_values": reader_values, 
            "summary": summary
            })
        print("\n\nEXPLAIN")
        print(result.content)

    if True:
        reader_values = """
        * Approaching software engineering pragmatically.
        * Unique creative approaches to common problems.
        * Mountain biking, running, hiking.
        """

        result = (ASSESS_WORTHINESS_PROMPT | llm).invoke({
            "reader_values": reader_values, 
            "summary": summary
            })
        print("\n\nWORTH READING?")
        print(result.content)
        assert check_text_entail_claim(result.content, "positive").is_true
        
        result = (EXPLAIN_WORTHINESS_PROMPT | llm).invoke({
            "reader_values": reader_values, 
            "summary": summary
            })
        print("\n\nEXPLAIN")
        print(result.content)
