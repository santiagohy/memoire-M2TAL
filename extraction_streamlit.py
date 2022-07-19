from ctypes.wintypes import PWIN32_FIND_DATAW
from fastapi import FastAPI, Form, Request, UploadFile, Cookie
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import grew
import rules4streamlit as rules

import streamlit as st
import pandas as pd




st.set_page_config(
    page_title="Rules extraction", page_icon="📐", initial_sidebar_state="expanded"
)

st.write(
    """
# 📐 Rules Extraction App
Upload your treebank to query grammar rules.
"""
)

uploaded_file = st.file_uploader("Upload CoNLL-U", type=".conllu")
if not uploaded_file:
    st.info('Please upload a file')
    st.stop()

with st.spinner('Loading treebank...'):
    grew.init()
    content = uploaded_file.read()
    treebank_idx, treebank = rules.load_corpus(content, uploaded_file.name)


with st.form("form"):
    st.write("Insert your querys")

    P1 = st.text_area("Pattern 1", value="e:X->Y; X[Number]; Y[Number]", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)

    # Solo con P1, se puede parsear para recuperar que son los traits disponibles...
    # If P1 : etc...

    P2 = st.text_area("Pattern 2", value="Y.Number = X.Number", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)
    P3 = st.text_area("Pattern 3", value="X.upos; e.label; Y.upos", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False)

    option = st.radio(
        "Mode",
        ('Largest combinations', 'All possible combinations'), horizontal=True)

    if option  == 'Largest combinations':
        option = False
    else:
        option = True
    
    submit = st.form_submit_button("Go!")


if submit:
    with st.spinner('Wait for it...'):
        df = rules.rules_extraction(treebank_idx, treebank, P1, P2, P3, option)
        st.dataframe(df)
else:
    st.info('Please complete the query patterns')
    st.stop()

