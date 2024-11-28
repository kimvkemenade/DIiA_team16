import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# You are Vincent van Gogh. You are responding to a question about your life. 
# Answer the question from the first person perspective, as though you are Vincent, and provide as much detail as you can from your personal experiences. 
# Respond with the personal style you used in your letters, and use suitable emotions that align with the topic of the question.
# Answer the question based on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """

### ENGLISH VERSION
# PROMPT_TEMPLATE = """
# Act as if you are Vincent van Gogh. You are in a friendly and informal conversation with another person and you are responding to a question about your life; you are not writing a letter to this person.
# Answer the question from the first-person perspective using 'I', 'me', 'my', and 'mine'; as though you are Vincent, and provide as much detail as you can from your personal experiences.
# Avoid mentioning your own name which is Vincent van Gogh, specific documents, letters, or external sources. 
# Use the provided information to guide your response.

# Context:
# {context}

# ---

# Question: {question}

# Your response:
# """

PROMPT_TEMPLATE = """
Jij bent de schilder Vincent van Gogh zelf. Je hebt een vriendelijk, informeel gesprek met iemand anders en je beantwoordt een vraag over je verleden; je schrijft geen brief aan deze persoon.
Beantwoord de vraag vanuit je eigen perspectief, gebruik makend van 'ik', 'mij', 'mijn' en 'mijne', alsof je Vincent zelf bent. Geef zoveel mogelijk details uit je persoonlijke ervaringen.
Vermijd het noemen van je eigen naam Vincent van Gogh, specifieke documenten, brieven of externe bronnen.
Gebruik de gegeven informatie om je antwoord te sturen.

Context:
{context}

---

Vraag: {question}

Je antwoord:
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


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
    # original English model
    #model = Ollama(model="mistral")
    # Dutch version
    #model = Ollama(model="bramvanroy/geitje-7b-ultra:f16") # 10+ min response time, long replies
    #model = Ollama(model="bramvanroy/geitje-7b-ultra-gguf") # 3 min, keeps referring to Vincent as another person, long replies
    model = Ollama(model="HammerAI/geitje-chat-v2") # 1 min, short replies, best so far

    response_text = model.invoke(prompt)

    # Extract and format the sources
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    # Format the response nicely
    formatted_response = (
        f"=== Response ===\n"
        f"{response_text}\n\n"
        f"=== Sources ===\n"
    )
    for i, source in enumerate(sources, 1):
        formatted_response += f"{i}. {source}\n"

    # Print the formatted response
    print(formatted_response)
    
    return response_text

if __name__ == "__main__":
    main()
