import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Web App
st.title("🎬 YouTube Transcription Generator")
title_input = st.text_input("What is the video about")

# Prompt
title_template = PromptTemplate(
    input_variables=["topic"],
    template="write me a shocking title for a YouTube video about {topic}"
)

# LLM
llm = OpenAI(temperature=0.7)

# Chain
title_chain = LLMChain(llm=llm, prompt=title_template)

# Show if title_input submitted
if title_input:
    response = title_chain.run(title_input)
    st.write(response)