#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import pipeline


# In[5]:


classifier = pipeline("sentiment-analysis")


# In[6]:


classifier("We are very happy to show you the 🤗 Transformers library.")


# In[7]:


results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


# In[14]:


import torch
from transformers import pipeline
print(torch.__version__)
print(torch.__file__)
speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")


# In[15]:


import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# In[16]:


from datasets import load_dataset, Audio
trust_remote_code=True
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")


# In[17]:


dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))


# In[20]:


result = speech_recognizer(dataset[:4]["audio"])
print([d["text"] for d in result])


# In[22]:


model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[23]:


classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")


# In[24]:


from transformers import AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# In[25]:


encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)


# In[26]:


pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)


# In[ ]:


from transformers import pipeline
# Speech recognition pipeline
transcriber = pipeline(task="automatic-speech-recognition")


# In[ ]:


transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")


# In[33]:


# Vision pipeline
from transformers import pipeline

vision_classifier = pipeline(model="google/vit-base-patch16-224")
preds = vision_classifier(
    images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
preds


# ![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg)
# 
# 

# In[ ]:


# text classification pipeline
from transformers import pipeline

# This model is a `zero-shot-classification` model.
# It will classify text, except you are free to choose any label you might imagine
classifier = pipeline(model="facebook/bart-large-mnli")
classifier(
    "I have a problem with my iphone that needs to be resolved asap!!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
)


# In[1]:


# Multimodal pipeline
from transformers import pipeline
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/Cellar/tesseract/5.5.0/bin/tesseract'

vqa = pipeline(model="impira/layoutlm-document-qa")
output = vqa(
    image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
    question="What is the invoice number?",
)
output[0]["score"] = round(output[0]["score"], 3)
output


# In[3]:


# pip install accelerate
import torch
from transformers import pipeline

pipe = pipeline(model="facebook/opt-1.3b", torch_dtype=torch.bfloat16, device_map="auto")
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)


# In[4]:


from transformers import pipeline
import gradio as gr

pipe = pipeline("image-classification", model="google/vit-base-patch16-224")

gr.Interface.from_pipeline(pipe).launch()

