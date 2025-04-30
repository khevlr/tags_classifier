import streamlit as st
from newspaper import Article
import torch
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import requests
from bs4 import BeautifulSoup
import time


model_path = "nurkz_bert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


with open('label_binarizer.pkl', 'rb') as f:
    classes = pickle.load(f)

def predict_tags(text, threshold=0.2):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: torch.tensor(val) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).numpy()[0]

    predicted_tags = [classes[i] for i, p in enumerate(probs) if p >= threshold]
    return predicted_tags

def get_text_tags(url, retries=3, delay=5):

    headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "en-US,en;q=0.9",
            }

    try:

        r = requests.get(url, timeout=(3,5), headers=headers)

        r.raise_for_status()

    except requests.exceptions.Timeout:

        print(f"Timeout при загрузке: {url}")
        
        if retries > 0:
            print(f"Повтор через {delay} сек... Осталось попыток: {retries}")
            time.sleep(delay)
            return get_text_tags(url, retries - 1, delay)
        else:
            return "", []
    
    except requests.exceptions.RequestException as e:

        print(f"Error downloading {url}: {e}")
        return "", []
    
    soup = BeautifulSoup(r.text, 'html.parser')
    
    paragraphs = soup.select('p.formatted-body__paragraph')

    article_text = ''''''

    for p in paragraphs:

        for el in p.find_all(['a', 'img', 'em', 'strong', 'span']):
            el.replace_with(f''' {el.get_text(strip=True)}''')

        text = p.get_text()

        if text:
            article_text += f''' {text}'''

    return article_text
    
st.title("📰 Предсказание тегов для новостей из nur.kz")
option = st.radio("Выберите способ ввода:", ("Текст вручную", "Ссылка на статью"))

text_input = ""
if option == "Текст вручную":
    text_input = st.text_area("Введите текст статьи:", height=300)
elif option == "Ссылка на статью":
    url = st.text_input("Введите URL статьи:")
    if url:
        text_input = get_text_tags(url)
        st.text_area("Извлечённый текст статьи:", text_input, height=300)

if st.button("Предсказать теги") and text_input.strip():
    tags = predict_tags(text_input)
    st.subheader("🏷️ Предсказанные теги:")
    if tags:
        st.write(", ".join(tags))
    else:
        st.write("Теги не определены (возможно, модель не уверена).")


