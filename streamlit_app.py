import streamlit as st
import pandas as pd
import numpy as np
import re
from keras.models import load_model
import spacy
from spacy.lang.ru.examples import sentences 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import configparser
import ru_core_news_sm
    
MAX_SEARCH_RESULTS=5
st.set_page_config(layout="wide")

st.title("В меру умный поиск по тексту")

def clear_text(text):
    clean_text = re.sub(r'(?:(?!\u0301)[\W\d_])+', ' ', text.lower())
    return clean_text

df_r = pd.read_json('RUSSE/train.jsonl', lines=True)
df_r = df_r[['sentence1']]
df_r= df_r.drop_duplicates()
df_r= df_r.reset_index(drop=True)
df_r['label_enc']= df_r.index

disabled_pipes = [ "parser",  "ner"]
#nlp = spacy.load('ru_core_news_sm', disable=disabled_pipes)
nlp= load(ru_core_news_sm)

lemmatizer = nlp.get_pipe('lemmatizer')
tokenizer = Tokenizer(oov_token='<oov>')

config = configparser.ConfigParser()
config.read('tokenizer.ini')
max_sequence_len= int(config['DEFAULT']['max_sequence_len'])
#st.write('max_sequence_len', max_sequence_len)
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
loaded_model= load_model('phrase_prediction_combo_rus.keras', compile = False)    
st.header('Данные датасета:')
update_df_sample = st.button('Обновить примеры данных')
placeholder = st.empty()
container = st.container()
if update_df_sample:
    with placeholder.container():
        st.write(df_r.sample(5))
with placeholder.container():
    st.write(df_r.sample(5)) 

#seed_text = "Он не"
seed_text = "если звонить мужу"
#seed_text = "Он не"
#seed_text = "Он не работать"
#seed_text = "В нашей"
#seed_text = "разговор с помощью ультразвука"

seed_text= st.text_input('Поисковый запрос', value=seed_text)
startSearch = st.button('Искать')
if startSearch:

    #st.write('seed_text:', seed_text)
    
    doc = nlp(clear_text(seed_text))
    lemm_text = " ".join([i.lemma_ for i in doc]) 
    #st.write('lemm_text:', lemm_text) 
    token_list = tokenizer.texts_to_sequences([lemm_text])[0]
    #st.write(f'token_list: {token_list}')
    
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')   
    #st.write(f'token_list padded: {token_list}')    
 
    #print(f'token_list padded: {token_list}')
    prediction= loaded_model.predict(token_list)
    #print('prediction:', prediction[0])
    x = np.argsort(prediction[0])[::-1][:MAX_SEARCH_RESULTS]#это N наиболее подходящих результатов (в том числе сюда попадают и те, которые не начинаются с заданной последовательности)
    x_m = np.sort(prediction[0])[::-1][:MAX_SEARCH_RESULTS]
    #st.write("Indices:",x)
    #st.write("Scores:",x_m.round(2))

    result_strings=[]
    result_scores=[]
    for result_index, score_val in zip(x, x_m):
        search_result = df_r[df_r['label_enc'] == result_index]
        result_strings.append(search_result['sentence1'].item())
        result_scores.append(round(score_val, 2))
    result_df= pd.DataFrame(columns=['Результат поиска', 'Релевантность'], data={'Результат поиска':result_strings, 'Релевантность': result_scores})
    st.header('Результаты поиска:')
    st.write(result_df)
   