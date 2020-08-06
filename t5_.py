import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
import pandas as pd
import re
import json 
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelWithLMHead.from_pretrained("t5-small")
device = torch.device('cpu')

def preprocess(text):
    preprocess_text1 = text.strip().replace("\n","")
    preprocess_text2 = preprocess_text1.lower()
    preprocess_text3 = re.sub(r'\d+', '', preprocess_text2)
    preprocess_text4 = preprocess_text3.translate(str.maketrans('', '', string.punctuation))
    # stop_words = set(stopwords.words("english")) 
    # word_tokens = word_tokenize(preprocess_text4) 
    # filtered_text = [word for word in word_tokens if word not in stop_words] 
    # preprocess_text = ' '.join(filtered_text)

    print("Text Cleaned!")
    return preprocess_text4

def textSum(text):
    output = ''

    t5_prepared_Text = "summarize: "+text
    print ("Inputted Text -- \n", text)

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

    summary_ids = model.generate(tokenized_text,
                                        num_beams=32,
                                        no_repeat_ngram_size=2,
                                        min_length=100,
                                        max_length=500,
                                        early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output
