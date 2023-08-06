import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# Web App
st.title("🎬 YouTube Transcription Generator")
prompt_input = st.text_input("What is the video about?")

# Prompts
title_template = PromptTemplate(
    input_variables=["topic"],
    template="write me a shocking title for a YouTube video about {topic}"
)

script_template = PromptTemplate(
    input_variables=["title", "wikipedia_research"],
    template="""write me a complete script for a YouTube video based on the title.
    %TITLE
    {title}

    while you leverage this Wikipedia research:
    {wikipedia_research}"""
)

# Memory
title_memory = ConversationBufferMemory(
    input_key="topic", memory_key="chat_history")
script_memory = ConversationBufferMemory(
    input_key="title", memory_key="chat_history")

# LLM
llm = OpenAI(temperature=0.7)

# Chains
title_chain = LLMChain(llm=llm, prompt=title_template,
                       output_key="title", memory=title_memory)

script_chain = LLMChain(llm=llm, prompt=script_template,
                        output_key="script", memory=script_memory)

wiki = WikipediaAPIWrapper()

# Show if prompt_input submitted
if prompt_input:
    title = title_chain.run(topic=prompt_input)
    wiki_research = wiki.run(prompt_input)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(script)

    with st.expander("Title Log"):
        st.info(title_memory.buffer)

    with st.expander("Script Log"):
        st.info(script_memory.buffer)

    with st.expander("Wikipedia Research"):
        st.info(wiki_research)
