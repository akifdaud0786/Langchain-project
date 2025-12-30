import validators
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------------- UI ----------------
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")


# ---------------- Prompt ----------------
prompt = ChatPromptTemplate.from_template("""
Provide a clear and concise summary of the following content in about 300 words.

Content:
{text}
""")


# ---------------- Button Logic ----------------
if st.button("Summarize the Content from YT or Website"):

    if not groq_api_key.strip():
        st.error("❌ Please enter your Groq API Key")

    elif not generic_url.strip():
        st.error("❌ Please enter a URL")

    elif not validators.url(generic_url):
        st.error("❌ Please enter a valid URL")

    else:
        try:
            with st.spinner("Summarizing..."):

                # ✅ Initialize LLM ONLY here
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model="openai/gpt-oss-20b"
                )

                chain = prompt | llm | StrOutputParser()

                # Load content
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=False
                        )

                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )

                docs = loader.load()

                combined_text = "\n\n".join(
                    doc.page_content for doc in docs
                )

                summary = chain.invoke({"text": combined_text})

                st.success(summary)

        except Exception as e:
            st.exception(e)
