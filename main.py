import streamlit as st
import pandas as pd
from model import *

st.title('Hello!')
st.markdown("<h1 style='text-align: center;'>Search the Most Matched Video<br>for You</h1>", unsafe_allow_html=True)

query = st.text_input('')
# documents = get_documents_dict()

if query:
    results = get_top_10_related(query)
    for i in results:
        id,title = get_id_title(i)
        st.markdown(f'<h3>{title}</h3>', unsafe_allow_html=True)
        st.markdown("https://www.youtube.com/watch?v={}".format(id), unsafe_allow_html=True)