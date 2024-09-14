from flask import Flask, request, jsonify, send_from_directory
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

# Initialize the model
model = OllamaLLM(model="llama3")

# Define the template for the conversation
template = """
Answer the question below.

Here is the conversation history: {context}

Question: {question}
Answer:
"""

# Create a prompt using the template
prompt = ChatPromptTemplate.from_template(template)

# Define the chain combining the prompt and model
chain = prompt | model

context = ""

@app.route('/ask', methods=['POST'])
def ask():
    global context
    data = request.json
    user_input = data.get('question')
    result = chain.invoke({"context": context, "question": user_input})
    result_str = str(result)
    context += f"\nUser: {user_input}\nAI: {result_str}"
    return jsonify({"answer": result_str})

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == "__main__":
    app.run(debug=True, port=5500)
