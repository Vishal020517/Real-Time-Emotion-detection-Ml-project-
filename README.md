**🎙️ Voice-Based Emotion Detection using Whisper & NLP**

This project combines speech recognition and natural language processing (NLP) to detect the emotion behind spoken words. Users can speak into the microphone, and the system will:

  1.Record your voice
  
  2.Transcribe your speech to text using OpenAI's whisper-small model via Hugging Face Transformers
  
  3.Analyze the emotion from the transcribed text using a trained NLP classifier

**Features**

  1.Live Voice Recording with sounddevice
  
  2.Speech-to-Text using OpenAI's Whisper (openai/whisper-small)
  
  3.Emotion Detection trained on the dair-ai/emotion dataset
  
  4.Text Preprocessing with NLTK (lemmatization, stopword removal, etc.)
  
  5.Logistic Regression Model for emotion classification

**Supports 6 basic emotions: sadness, joy, love, anger, fear, surprise**


Example Workflow
  1.Run the script
  
  2.Speak something like: "I'm feeling great today!"
  
  3.Output:Transcribed Text: I'm feeling great today!
           Predicted Emotion: joy
           
