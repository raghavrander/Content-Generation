import streamlit as st
from  text import generate
length = st.text_input('Text length required:', '')
existing_content = st.text_input('Enter your content here:', '')
if(st.button("Generate")):
    text = generate(existing_content,int(length))
    # text = text.pop()
    for i in text:
        st.write(i)
