import torch
from datasets import load_dataset
from transformers import pipeline

def tts_model():

  model = pipeline("text-to-speech", "microsoft/speecht5_tts", device=0)

  dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
  embedding = torch.tensor(dataset[7306]["xvector"]).unsqueeze(0)

  return model, embedding

def speak(model, embedding, text):

  speech = model(text, forward_params={"speaker_embeddings": embedding})
  return speech