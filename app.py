import streamlit as st
import os
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the GROQ API Key
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Groq API Key is missing.")
    st.stop()  # Stop execution if key is missing

# Initialize the ChatGroq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.2-1b-preview")

# Streamlit UI setup
st.title("🌐 Competitive Programming Solver")
st.write("**Get your optimized Python solution by providing a problem statement!**")

# Initialize session state for the problem statement and solution
if 'problem_statement' not in st.session_state:
    st.session_state.problem_statement = ""
if 'generated_solution' not in st.session_state:
    st.session_state.generated_solution = ""

# Input field for the problem statement
problem_statement = st.text_area(
    "Enter the Competitive Programming Problem Statement:",
    placeholder="Provide the full description of the problem here..."
)

# Button to generate solution
if st.button("Generate Solution"):
    if not problem_statement.strip():
        st.error("Please enter a valid problem statement.")
    else:
        # Save the problem statement to session state
        st.session_state.problem_statement = problem_statement

        # Create the dynamic prompt text, requesting only the code output
        prompt_text = f"""
            You are given the following competitive programming problem:

            {st.session_state.problem_statement}

            Your task is to:
            1. Provide only the most optimized Python solution.
            2. Return only the code without any additional text, comments, or explanations.
        """

        # Invoke the model with the prompt text
        response = llm.invoke(prompt_text)

        # Extract text from response object if possible
        if hasattr(response, 'text'):
            solution = response.text.strip()  # Extract text and strip white spaces
        else:
            solution = str(response).strip()  # Convert to string if necessary

        # Save the solution to session state
        st.session_state.generated_solution = solution if solution else "Failed to generate a solution."

# Display the stored problem statement and generated solution
if st.session_state.problem_statement:
    st.write("### Problem Statement:")
    st.write(st.session_state.problem_statement)

if st.session_state.generated_solution:
    st.write("### Generated Solution:")
    st.code(st.session_state.generated_solution, language='python')

# Feedback section
feedback = st.text_input("Provide feedback on the solution (correct/incorrect):")
if feedback:
    st.write("Thank you for your feedback!")
    # Additional feedback handling logic can be implemented here
