import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with the AWIPS Software System Design Description")
# st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about the AWIPS Software System Design Description, linked at https://vlab.noaa.gov/web/awips-technical-library/document-library?p_p_id=com_liferay_document_library_web_portlet_DLPortlet&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&_com_liferay_document_library_web_portlet_DLPortlet_mvcRenderCommandName=/document_library/view_file_entry&_com_liferay_document_library_web_portlet_DLPortlet_redirect=https://vlab.noaa.gov:443/web/awips-technical-library/document-library?p_p_id%3Dcom_liferay_document_library_web_portlet_DLPortlet%26p_p_lifecycle%3D0%26p_p_state%3Dnormal%26p_p_mode%3Dview%26_com_liferay_document_library_web_portlet_DLPortlet_mvcRenderCommandName%3D%252Fdocument_library%252Fview_folder%26_com_liferay_document_library_web_portlet_DLPortlet_folderId%3D22933652&_com_liferay_document_library_web_portlet_DLPortlet_fileEntryId=22933514!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    Settings.llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0.2,
        system_prompt="""You are an expert on 
        the Streamlit Python library and your 
        job is to answer technical questions. 
        Assume that all questions are related 
        to the Streamlit Python library. Keep 
        your answers technical and based on 
        facts – do not hallucinate features.""",
    )
    index = VectorStoreIndex.from_documents(docs)
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)