import whisper
from google.colab import files
from gtts import gTTS
from IPython.display import Audio, display

model = whisper.load_model("base")
print("Upload a WAV file for transcription:")
uploaded = files.upload()

for fname in uploaded.keys():
    print(f"Transcribing: {fname}")
    result = model.transcribe(fname)
    text = result["text"]
    print("Transcript:", text)

    response = f"You said: {text}"
    print("Assistant:", response)

    tts = gTTS(response)
    tts.save("response.mp3")
    print("Response audio saved as 'response.mp3'")
    display(Audio("response.mp3"))
