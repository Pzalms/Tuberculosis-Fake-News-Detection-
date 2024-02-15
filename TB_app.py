import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
import base64

st.set_page_config(layout="wide", page_title="Tuberculosis Fake News Detection")

st.title(":black[TB Fake News Detection]")
st.subheader("Web App to Detect TB Fake News Using Support Vector Machine Model")

Text = st.text_input(label='Enter your text here')
       
if st.button("**Classify Text**"):
    result = predict(Text)
    st.success("**:green[{}]**".format(result))