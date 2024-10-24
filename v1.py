import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import torch
import time
from dotenv import load_dotenv

load_dotenv()

# Load Llama model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to generate response using Llama model
def generate_llama_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Session management: Store chat history and feedback
if "history" not in st.session_state:
    st.session_state.history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []

# Prompt for competitive programming
prompt_template = """
The user will provide a link to a competitive programming question. Your task is to read the problem from the given link and provide an optimized solution in Python.

Your response should include:
- The Python code solving the problem in the most optimized way.
- A brief explanation of the approach used to solve the problem, including any optimizations.
- Ensure that the code is correct, adheres to the constraints, and passes all edge cases mentioned in the problem description.

Guidelines:
- Prioritize correctness first, ensuring the solution satisfies the problem's constraints and expected input/output formats.
- Optimize the code for efficiency, minimizing time and space complexity as much as possible.
- If multiple approaches exist, choose the most efficient one and explain why it was chosen.
- Ensure the solution works for large input sizes and extreme cases, as competitive programming problems often require optimized solutions.

Problem URL: {input}
"""

# Function to scrape website content
def scrape_website_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from paragraphs and combine
        paragraphs = soup.find_all('p')
        text_content = ' '.join([para.get_text() for para in paragraphs])

        return text_content

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching website content: {e}")
        return None

# Create vector embeddings from website content
def create_vector_embedding_from_url(url):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Scrape website and get the text
        website_content = scrape_website_content(url)
        if website_content:
            # Split the content into documents (chunks)
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.create_documents([website_content])
            
            # Create vector store
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        else:
            st.error("Failed to retrieve content from the website")

st.title("🌐 Competitive Programming Q&A with Llama-3.2-1B")
st.write("**Get Python solutions to competitive programming questions.**")

url_input = st.text_input("Enter a problem URL", value="https://example.com/problem")
user_prompt = st.text_input("Enter your query or ask for a solution", placeholder="Solve this problem...")

if st.button("Generate Embedding"):
    create_vector_embedding_from_url(url_input)
    st.write("Vector Database is ready")

# Handle feedback from the user
if st.session_state.history:
    st.write("### Chat History")
    for idx, item in enumerate(st.session_state.history):
        st.write(f"**User:** {item['user']}")
        st.write(f"**Model:** {item['model']}")

        # Display feedback options
        feedback = st.radio(f"Was the solution correct? (For response {idx+1})", ("Yes", "No"), key=f"feedback_{idx}")
        if feedback != st.session_state.feedback[idx]:
            st.session_state.feedback[idx] = feedback

# Generate a response from the Llama model based on user input
if user_prompt:
    prompt = prompt_template.format(input=user_prompt)
    with st.spinner("Generating solution..."):
        response = generate_llama_response(prompt)
    
    # Append the interaction to the chat history
    st.session_state.history.append({"user": user_prompt, "model": response})
    st.session_state.feedback.append("Pending")  # Initialize feedback as pending

    st.write(f"**Model's Response:**\n\n{response}")

# Optionally display detailed feedback from the user
if st.session_state.feedback:
    st.write("### User Feedback Summary")
    for i, feedback in enumerate(st.session_state.feedback):
        st.write(f"Response {i+1}: {feedback}")
