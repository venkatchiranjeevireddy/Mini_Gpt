import wikipedia
from serpapi import GoogleSearch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# === Config ===
SERP_API_KEY = "ff1149fce4e8c2fba6fce804621efed95bd309630af39ed4ec734d5002b7"
MODEL_NAME = "microsoft/phi-2"
# === Tools ===
def clean_query(query):
    return query.lower().replace("what is mean by", "").replace("?", "").strip()
def get_wikipedia_summary(query):
    try:
        wikipedia.set_lang("en")
        return wikipedia.summary(query, sentences=3)
    except:
        return "No reliable Wikipedia summary found."
def get_google_snippets(query):
    try:
        search = GoogleSearch({"q": query, "api_key": SERP_API_KEY})
        results = search.get_dict().get("organic_results", [])
        return " ".join([res["snippet"] for res in results if "snippet" in res][:3])
    except:
        return "No Google results found."
def generate_answer(query, wiki, google):
    prompt = f"""
You are a helpful AI assistant. Answer the user's question clearly and concisely, using the information provided.
Wikipedia: {wiki}
Google: {google}
Question: {query}
Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean response
    answer = full_output.split("Answer:")[-1].split("Question:")[0].strip()
    return answer
# === Main loop ===
def ask_question(query):
    q_clean = clean_query(query)
    print("Wikipedia...")
    wiki = get_wikipedia_summary(q_clean)
    print("Google...")
    google = get_google_snippets(q_clean)
    print("Phi-2 thinking...\n")
    final = generate_answer(query, wiki, google)
    print("Final Answer:\n" + final)
# === Run ===
if __name__ == "__main__":
    while True:
        q = input("\n Ask a question (or type 'exit'): ")
        if q.lower().strip() == "exit":
            print("Bye!")
            break
        ask_question(q)
