
import streamlit as st
import pickle
import nltk
import re
from nltk.stem import PorterStemmer
stopwords=nltk.corpus.stopwords.words('english')



model=pickle.load(open('D:\\sentiment_analysis\\logistic.pkl','rb'))
tfidf=pickle.load(open('D:\\sentiment_analysis\\tfidf.pkl','rb'))

st.title("Emotion Analysis")


def clean_text(text):
  stemmer=PorterStemmer()
  text=re.sub("[^a-zA-Z]"," ",text)
  text=text.lower()
  text=text.split()
  text=[stemmer.stem(word) for word in text if word not in stopwords]
  return " ".join(text)

def predict_emotion(text_):
  cleaned_text=clean_text(text_)
  tfidf_=tfidf.transform([cleaned_text])
  predict_label=model.predict(tfidf_)[0]
  return predict_label

text__=st.text_input("write your text here__")

if st.button("prediction"):
  emo=predict_emotion(text__)
  if emo==0:
    st.text("sad")
  if emo==1:
    st.text("joy")
  if emo==2:
    st.text("love")
  if emo==3:
    st.text("angry")
  if emo==4:
    st.text("fear")
  if emo==0:
    st.text("surprise")                