from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage

def test_it_works():
    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype="auto",
    )

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        return_full_text=False,
        max_new_tokens=8192,
        pad_token_id=tok.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=gen_pipe)
    chat = ChatHuggingFace(llm=llm)

    def render(messages):
        role_map = {"human": "user", "ai": "assistant", "system": "system"}
        hf_msgs = [{"role": role_map[m.type], "content": m.content} for m in messages]
        return tok.apply_chat_template(hf_msgs, tokenize=False, add_generation_prompt=True)

    messages = [
        SystemMessage("You are concise and helpful.")
    ]

    def say(message):
        print("***")
        print("User:", message)
        messages.append(HumanMessage(message))
        prompt = render(messages)
        resp = chat.invoke(prompt)
        print(f"Assistant:", resp.content)
        return resp.content
        
    assert("sandwiches" in say("say 'sandwiches'").lower())
    assert("5" in say("add two and three. respond with a single number, no punctuation"))
