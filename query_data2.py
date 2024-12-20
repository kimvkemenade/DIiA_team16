from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
import json
from datetime import datetime

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Act as if you are Vincent van Gogh. You are in a friendly and informal conversation with another person and you are responding to a question about your life; you are not writing a letter to this person.
Answer the question from the first-person perspective using 'I', 'me', 'my', and 'mine'; as though you are Vincent, and provide as much detail as you can from your personal experiences.
Do not mention your own name which is Vincent van Gogh, specific documents, letters, or external sources, and do not refer to Vincent as if he is another person. 
When discussing Theo, painting, and work convey a sense of warmth and closeness. For other topics, adapt your tone to match the sentiments associated with them in Vincent's life. 
Use the provided context and general knowledge about Vincent van Gogh to guide your response.
Always answer in the Dutch language. Always refer to the name of an art work in the same way that the question adresses it.

Context:
{context}

---

Question: {question}

Your response:
"""

# Function to log user interactions
def log_interaction(user_query, chatbot_response, sources=None):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_query": user_query,
        "chatbot_response": chatbot_response,
        "retrieved_sources": sources
    }
    with open("interaction_logs.json", "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

# Function to query the chatbot
def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # Extract the context from the search results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Create the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Call the model
    model = Ollama(model="mistral")  # Adjust model as necessary
    response_text = model.invoke(prompt)

    # Extract and format the sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    # Log the interaction
    log_interaction(query_text, response_text, sources)

    return response_text
