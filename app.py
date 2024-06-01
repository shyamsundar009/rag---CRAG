import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from crag import main, aws, file_to_chunks,azure_data_download
import os
st.title("CRAG Q/A")


# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Which data source you are using?",
    ("Azure Blob Storage", "Local storage Injest","AWS S3 Bucket","Locally stored Vectorstore DB"),
    index=None,
    placeholder="Select contact method...",
)

if add_selectbox == "Azure Blob Storage":
    st.sidebar.write("You selected Azure Blob Storage.")
    AZURE_CONNECTION_STRING = st.sidebar.text_input("Azure Connection String Input",type="password")
    CONTAINER_NAME = st.sidebar.text_input("Azure Container Name")
elif add_selectbox == "Local storage Injest":
    st.sidebar.write("You selected Local storage Injest.")
    uploaded_files = st.sidebar.file_uploader("Choose the files for Injest", accept_multiple_files=True)
elif add_selectbox == "AWS S3 Bucket":
    aws_access_key = st.sidebar.text_input("AWS Access Key",type="password")
    aws_secret_access_key = st.sidebar.text_input("AWS SECRET ACCESS KEY",type="password")
    bucket_name= st.sidebar.text_input("AWS BUCKET NAME")
    object_name= st.sidebar.text_input("AWS OBJECT NAME")
elif add_selectbox == "Locally stored Vectorstore DB":
    vectorstore_name = st.sidebar.text_input("Vectorstore db Name")
else:
    st.sidebar.write("You selected nothing.")

if st.sidebar.button("Injest"):
    if add_selectbox == "Azure Blob Storage" and AZURE_CONNECTION_STRING and CONTAINER_NAME:
        # Download PDF from Azure Blob Storage
        with st.sidebar:
            try:
                with st.spinner("Azure connection is creating....."):
                    azure_data_download(AZURE_CONNECTION_STRING=AZURE_CONNECTION_STRING, CONTAINER_NAME=CONTAINER_NAME)
                with st.spinner("Azure Folder documents to chunks are in the process.........."):
                    pages = file_to_chunks("Azure_data")
                with st.spinner("VectorDatabse is creating....."):
                    db = FAISS.from_documents(pages, OpenAIEmbeddings())
                    db.save_local("Azure_Chroma_db")
                    st.session_state.location = "Azure_Chroma_db"
                st.success("VectorDatabse is created successfully in the Azure_Chroma_db")
            except Exception as e:
                st.error(f"Error connecting with Azure: {str(e)}")
    elif add_selectbox == "Local storage Injest" and uploaded_files:
        if not os.path.exists("Local_data"):
            os.makedirs("Local_data")
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                # Save file in the specified directory
                with open(os.path.join("Local_data", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer()) 
        with st.sidebar:
            with st.spinner("Local Folder documents to chunks are in the process.........."):
                pages = file_to_chunks("Local_data")   
            with st.spinner("VectorDatabse is creating....."):
                db = FAISS.from_documents(pages, OpenAIEmbeddings())
                db.save_local("Local_vectorstore")
                st.session_state.location = "Local_vectorstore"
        #     st.success("VectorDatabse is created successfully in the Local_vectorstore folder")
    elif add_selectbox == "Locally stored Vectorstore DB":
        if vectorstore_name=="db_001":
            with st.sidebar:
                st.session_state.location = "db_001"
                st.success("VectorDatabse is loaded successfully from the Local_vectorstore folder")
        else:
            st.warning(f"There is no vectorstore with name {vectorstore_name} in the Current folder")
    else:
        with st.sidebar:
            try:
                with st.spinner("AWS connection is creating....."):
                    aws(AWS_ACCESS_KEY_ID=aws_access_key, AWS_SECRET_ACCESS_KEY=aws_secret_access_key, BUCKET_NAME=bucket_name, object_name=object_name)
                with st.spinner("AWS Folder documents to chunks are in the process.........."):
                    pages = file_to_chunks("S3_data")
                with st.spinner("VectorDatabse is creating....."):
                    db = FAISS.from_documents(pages, OpenAIEmbeddings())
                    db.save_local("S3_data db")
                    st.session_state.location = "S3_data db"
                st.success("VectorDatabse is created successfully in the AWS_Chroma_db folder")
            except Exception as e:
                st.error(f"Error in connecting with AWS: {str(e)}")

    st.session_state.injest = True
    # st.session_state


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "injest" not in st.session_state:
    st.session_state.injest = False
if "location" not in st.session_state:
    st.session_state.location = ""


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Lets have a chat with our document")

if prompt and st.session_state.injest:
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Generating response by using CRAG..."):
        response = main(question=prompt,loc=st.session_state.location)
    
    with st.chat_message("assistant"):
        st.markdown(f"{response}")
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})