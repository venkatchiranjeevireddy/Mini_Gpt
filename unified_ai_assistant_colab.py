# unified_ai_assistant_colab.py (Optimized for Colab GPU with low resources)

import os
import wikipedia
import torch
import requests
from io import BytesIO
from PIL import Image
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline,
    BlipProcessor, BlipForConditionalGeneration
)
from huggingface_hub import login
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from IPython.display import Audio, display
from gtts import gTTS
from google.colab import files
from serpapi.google_search import GoogleSearch # Corrected import
#Config 
SERP_API_KEY = "ff1149fce4e8c2fba6fce804621efed95bd309630af39ed4ec9d474d5002b7"
HUGGINGFACE_TOKEN = "hf_RCPHowCTgRJlmGPTXmhkLPyUcHbArsg"
login(token=HUGGINGFACE_TOKEN)
# Paths to models
LLAMA_MODEL_PATH = "meta-llama/Llama-2-7b-hf"
FT_CHAT_PATH = "/content/drive/MyDrive/all_paths/lora_output/final_lora_adapter"
FT_SUMMARY_PATH = "/content/drive/MyDrive/all_paths/llama2-lora-xsum-checkpoints_summarization/checkpoint-5000"
FT_TEXTGEN_PATH = "/content/drive/MyDrive/all_paths/lora-llama2-gen_text/checkpoint-647"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
llm_pipelines = {}
image_pipe = None
blip_processor, blip_model = None, None
whisper_model = None
def load_finetuned_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        quantization_config=bnb_config
    )
    return tokenizer, model
def preload_models():
    global llm_pipelines, image_pipe, blip_processor, blip_model, whisper_model
    for task, path in zip(["chat", "summarize", "textgen"], [FT_CHAT_PATH, FT_SUMMARY_PATH, FT_TEXTGEN_PATH]):
        try:
            tokenizer, model = load_finetuned_model(path)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
            llm_pipelines[task] = pipe
        except Exception as e:
            print(f"Error loading {task} model: {e}")
    from diffusers import StableDiffusionPipeline
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        image_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        ).to("cuda")
    else:
        print("Image generation requires CUDA GPU with compute capability >= 7.0")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")
    import whisper
    whisper_model = whisper.load_model("base")
def hf_chat_loop(task):
    pipe = llm_pipelines.get(task)
    if not pipe:
        print("Pipeline not loaded.")
        return
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        prompt = q if task == "chat" else f"{task.title()} this:\n{q}\nOutput:"
        out = pipe(prompt)[0]['generated_text']
        print(" ", out.split("Output:")[-1].strip())
def live_retrieval():
    tokenizer, model = load_finetuned_model(LLAMA_MODEL_PATH)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
    def search(query):
        try:
            wiki = wikipedia.summary(query, sentences=2)
        except:
            wiki = "Wikipedia not found."
        try:
            results = GoogleSearch({"q": query, "api_key": SERP_API_KEY}).get_dict()["organic_results"]
            google = " ".join([r['snippet'] for r in results if 'snippet' in r][:3])
        except:
            google = "Google not found."
        return wiki, google
    while True:
        q = input("Ask a question (or exit): ")
        if q.lower() == "exit": break
        wiki, google = search(q)
        prompt = f"""Wikipedia: {wiki}
Google: {google}
Question: {q}
Answer:"""
        out = pipe(prompt)[0]['generated_text']
        print(" ", out.split("Answer:")[-1].strip())
def document_qa():
    def load_doc(fp):
        ext = os.path.splitext(fp)[1].lower()
        if ext == ".pdf": return PyPDFLoader(fp).load()
        if ext == ".txt": return TextLoader(fp).load()
        if ext == ".docx": return UnstructuredWordDocumentLoader(fp).load()
        if ext == ".csv": return CSVLoader(fp).load()
        raise ValueError("Unsupported file")
    file_path = input("Enter doc path: ").strip()
    docs = load_doc(file_path)
    db_path = os.path.splitext(os.path.basename(file_path))[0] + "_faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    if os.path.exists(db_path):
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(db_path)
    tokenizer, model = load_finetuned_model(LLAMA_MODEL_PATH)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    llm = HuggingFacePipeline(pipeline=pipe)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
    while True:
        q = input("❓ You: ")
        if q.lower() == "exit": break
        print(" ", qa.invoke(q))
def image_gen():
    global image_pipe
    if not image_pipe:
        print(" Image generation pipeline not loaded.")
        return
    print(" Image generation model ready.")
    while True:
        prompt = input(" Describe image (or exit): ")
        if prompt.lower() == "exit": break
        img = image_pipe(prompt).images[0]
        img.save("generated_image.png")
        display(img)
def image_text_qa():
    global blip_processor, blip_model
    while True:
        url = input("📸 Enter image URL (or exit): ")
        if url.lower() == "exit": break
        image = Image.open(BytesIO(requests.get(url).content)).convert('RGB')
        inputs = blip_processor(images=image, return_tensors="pt").to(blip_model.device)
        out = blip_model.generate(**inputs)
        print(" Caption:", blip_processor.decode(out[0], skip_special_tokens=True))
def speech_chat():
    global whisper_model
    print(" Upload a WAV file to transcribe...")
    uploaded = files.upload()
    for fname in uploaded.keys():
        print(f" Transcribing: {fname}")
        result = whisper_model.transcribe(fname)
        text = result["text"]
        print("You said:", text)
        response = f"You said: {text}"
        print(" Assistant:", response)
        tts = gTTS(response)
        tts.save("response.mp3")
        print(" Saved as response.mp3")
        display(Audio("response.mp3"))

def main():
    preload_models()
    print("""
  Choose a mode:
1.   Chat
2.   Summarization
3.   Document Generation
4.   Document Q&A
5.   Live Wikipedia + Google Retrieval
6.   Image Generation
7.  Image Captioning / VQA
8.  Speech Chatting
""")
    mode = input("Select [1-8]: ").strip()
    if mode == "1": hf_chat_loop("chat")
    elif mode == "2": hf_chat_loop("summarize")
    elif mode == "3": hf_chat_loop("textgen")
    elif mode == "4": document_qa()
    elif mode == "5": live_retrieval()
    elif mode == "6": image_gen()
    elif mode == "7": image_text_qa()
    elif mode == "8": speech_chat()
    else: print("Invalid choice.")
if __name__ == "__main__":
    main()
