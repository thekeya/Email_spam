import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import string
import sklearn
char = string.punctuation
ps = PorterStemmer()
def all_preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', char))
    text = nltk.word_tokenize(text)
    list = []
    for i in text:  # stopword removal
        if i not in stopwords.words('english'):
            list.append(i)
    list = " ".join(list)
    x = []
    for word in list.split():  # stemming
        x.append(ps.stem(word))

    return " ".join(x)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/sms spam classifier")
input_sms = st.text_input("enter the message")
transformed_sms = all_preprocess(input_sms)
# st.title(transformed_sms)
vector_input = tfidf.transform([transformed_sms])
result = model.predict(vector_input)[0]
if st.button('Predict'):
       if ( result == 1):
                         st.header("Spam")
       else:
                         st.header("Not Spam")
