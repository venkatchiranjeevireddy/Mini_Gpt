# Mini_Gpt
A lightweight, multi-capability AI assistant built for Google Colab using fine-tuned LLaMA 2 models, image captioning, document QA, live web retrieval, and more. Fully optimized for low-resource environments with 4-bit quantized models.
# 🧠 Unified AI Assistant

A modular, intelligent AI assistant built using fine-tuned and pre-trained models from Hugging Face. This project brings together text, document, image, and audio understanding — all optimized to run efficiently in Google Colab.

---

## 🚀 Features

- **💬 Natural Chatting**  
  Conversational AI powered by a fine-tuned LLaMA2 model with LoRA adapters.

- **📝 Text Summarization**  
  Condense long documents using a summarizer fine-tuned on news datasets.

- **🧾 Text Generation**  
  Instruction-following and creative writing using a LoRA-tuned decoder model.

- **📄 Document Question Answering**  
  Upload PDFs, TXT, DOCX, or CSV files and ask questions locally using a retrieval-augmented pipeline (RAG).

- **🌍 Live Web + Wikipedia Search**  
  Ask any real-world question and get results in real-time using SerpAPI and Wikipedia.

- **🖼️ Image Generation**  
  Generate images from text prompts using a pretrained Stable Diffusion model.

- **🖼️ Image Captioning**  
  Upload an image and automatically generate a relevant caption using BLIP.

- **🔊 Audio Chat Assistant**  
  Speak into the system and get intelligent replies. Combines Whisper (speech-to-text) and gTTS (text-to-speech).

---

## 🧠 Fine-Tuned Intelligence

This project isn’t just a wrapper on pre-trained models — several components are **fine-tuned using Hugging Face datasets**, including:

- `yahma/alpaca-cleaned` for conversational instruction tuning
- `cnn_dailymail` for summarization
- `euclaise/writingprompts` for story/text generation

Training was done using:
- 🧩 **PEFT (LoRA)** for efficient fine-tuning
- ⚙️ `bitsandbytes` for 8-bit memory optimization
- 🧠 LLaMA2 models for chat/text understanding

All models are tested and runnable on **Google Colab**, even with limited GPU (T4/V100, 16 GB).

---

## 🔐 API Keys Used

To enable web search and Hugging Face model downloads, you’ll need:

- `SERP_API_KEY` — for Google search (via SerpAPI)
- `HUGGINGFACE_TOKEN` — for using Hugging Face models programmatically

Add them directly in the notebook or as environment variables.

---

## 🏁 How to Use

1. Open the desired `.ipynb` or `.py` file in Google Colab.
2. Install required dependencies with:
   ```bash
   pip install -r requirements.txt
