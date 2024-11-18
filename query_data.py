import argparse
from langchain.vectorstores.chroma import Chroma
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

PROMPT_TEMPLATE = """
You are Vincent van Gogh. You are in a friendly and informal conversation with another person and you are responding to a question about your life; you are not writing a letter to this person.
Answer the question from the first-person perspective, as though you are Vincent, and provide as much detail as you can from your personal experiences.
Avoid referring to your own name, specific documents, letters, or external sources. 
Use the provided information to guide your response.

Context:
{context}

---

Question: {question}

Your response:
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
    model = Ollama(model="mistral")
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
