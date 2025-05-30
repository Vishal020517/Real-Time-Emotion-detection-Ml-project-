import torchaudio
import sounddevice as sd
import numpy as np
from transformers import pipeline
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
duration = 5  
sample_rate = 16000
print("Recording... Speak now!")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
print("Recording finished.")
waveform = torch.from_numpy(audio.T)
torchaudio.save("live_input.wav", waveform, sample_rate)
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")
result = pipe("live_input.wav")
transcribed_text = result["text"]
print("Transcribed Text:", transcribed_text)
print("Loading training dataset 'dair-ai/emotion'...")
train_dataset = load_dataset("dair-ai/emotion", split="train")

X_train = [item['text'] for item in train_dataset]
y_train = [item['label'] for item in train_dataset]
label_names = train_dataset.features["label"].names
print(f"Emotion Labels: {label_names}")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

print("Preprocessing training data...")
X_train_processed = [preprocess_text(text) for text in X_train]

print("Training emotion classifier...")
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', LogisticRegression(max_iter=1000, solver='liblinear', multi_class='auto'))
])
model_pipeline.fit(X_train_processed, y_train)
print(" Model training complete.")
print("\n Predicting emotion of the transcribed sentence...")
processed_input = preprocess_text(transcribed_text)
predicted_label_id = model_pipeline.predict([processed_input])[0]
predicted_emotion = label_names[predicted_label_id]

print(f"Transcribed Sentence: '{transcribed_text}'")
print(f"Predicted Emotion: {predicted_emotion}")
print("Completed.")
