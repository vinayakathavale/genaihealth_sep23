import openai
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, NLTKTextSplitter
# import nltk
# nltk.download('punkt')
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from IPython.display import display, Markdown
from langchain.text_splitter import SpacyTextSplitter


openai.api_key = "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"
dir_path = "/Users/vinayak/code/genaihealth_sep23/data"


# PDF loader: extracts text data
def load_pdfs(dir_path):

    dir_loader = DirectoryLoader(dir_path, glob="**/*.pdf",
                            loader_cls=PyPDFLoader,
                            show_progress=True,
                            # use_multithreading=True,
                            silent_errors=True)
    docs = dir_loader.load()
    print(f"\nNumber of docs after initial loading: {len(docs)}, from: {dir_path}")

    return docs

## Chunk text data from docs

def chunk_docs(docs, chunk_size=2000, nltk=False, spacy=False, recursive=False):

    if nltk:
        text_splitter = NLTKTextSplitter(chunk_size=chunk_size,
                                         chunk_overlap=0)

    elif spacy:
        text_splitter = SpacyTextSplitter(chunk_size=chunk_size)

    elif recursive:
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=0,
                length_function=len,
                separators=["\n\n\n","\n\n", "\n", ".", " ", ""],)

    else:
        text_splitter = CharacterTextSplitter(
                                separator='\n',
                                chunk_size=chunk_size,
                                chunk_overlap=0,
                                length_function=len,)

    all_text = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    chunks = text_splitter.create_documents(all_text, metadatas=metadatas)
    print(f"Number of chunks: {len(chunks)}")

    return chunks


def add_chunk_index(chunks):

    sources = []
    for chunk in chunks:
        sources.append(chunk.metadata['source'])
    list(set(sources))
    for source in sources:
        chunk_index = 0
        for chunk in chunks:
            if source == chunk.metadata['source']:
                chunk.metadata['chunk_index'] = chunk_index
                chunk_index += 1
            else:
                continue
        total_chunks = chunk_index
        for chunk in chunks:
            if source == chunk.metadata['source']:
                chunk.metadata['last_chunk_index'] = total_chunks - 1

    print(f"Added chunk_order to metadata of {len(chunks)} chunks")

    return chunks

def get_chroma_vectorstore(chunks,
                            use_openai=True,
                            persist_directory="chroma_db"):

    if use_openai:
        model_name = 'text-embedding-ada-002'
        embeddings = OpenAIEmbeddings(model=model_name,
                                      openai_api_key=openai.api_key)
    else:
        model_name = "hkunlp/instructor-xl"
        embed_instruction = "Represent the text from the clinical guidelines"
        query_instruction = "Query the most relevant text from clinical guidelines"
        embeddings = HuggingFaceInstructEmbeddings(model_name=model_name,
                                                   embed_instruction=embed_instruction,
                                                   query_instruction=query_instruction)

    # Create vectorstore
    vectorstore = Chroma.from_documents(documents=chunks,
                                        embedding=embeddings,
                                        persist_directory=persist_directory)

    vectorstore.persist()

    print(f"Created chroma vectorscore, called: {persist_directory}")

    return vectorstore


def get_top_results_with_scores(query, vectorstore, k=4, threshold=0.4):

    results_with_scores = vectorstore.similarity_search_with_relevance_scores(query=query,
                                                                          k=k)
    score_threshold = threshold
    results_with_scores = [(doc, similarity)
                            for doc, similarity in results_with_scores
                           if similarity <= score_threshold]

    print(f"Number of results returned: {len(results_with_scores)}")

    return results_with_scores


def get_completion(prompt, model="gpt-3.5-turbo-16k", temperature=0.05):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]


### Chat function:

from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


def chat(user_input, vectorstore):


    llm = ChatOpenAI(openai_api_key="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")

    info = """
    Patient: Mrs Smith
    The patient is to undergo an elective total hip replacement operation, under regional anaesthetic (not general anaesthetic)
    10 days to go until the operation
    """

    results_with_scores = get_top_results_with_scores(query=user_input,
                                    vectorstore=vectorstore,
                                    k=2, threshold=0.8)

    guideline_info = ""
    chunks_seen = []
    pages_seen = []
    for result, score in results_with_scores:
        guideline_info += result.page_content + "\n\n"
        chunks_seen.append(result.metadata['chunk_index'])
        pages_seen.append(result.metadata['page'])

    system_message = f"""
You are a helpful assistant, assisting a patient with their elective surgery operation journey.
You can answer the questions they have regarding the operation using the information provided.

Here is the general information about the patient and operation:
{info}

Only use this information from the patient information leaflet to answer the patient's question:
{guideline_info}

Finish your output with the following:
"Do you have any further questions? Is there anything else I can help you with?"
"""


    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                system_message
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name.
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory
    )

    output = conversation({"question": user_input})

    return output, pages_seen, chunks_seen

def preprocess_index_db():
    docs = load_pdfs(dir_path=dir_path)
    all_chunks = chunk_docs(docs, recursive=True)
    print(f"Size of chunks combined: {len(all_chunks)}")

    all_chunks = add_chunk_index(all_chunks)

    for chunk in all_chunks:
        print(chunk.metadata)
        break

    persist_directory = "chroma_db"
    vectorstore_ = get_chroma_vectorstore(all_chunks,
                                            use_openai=True,
                                            persist_directory=persist_directory)
    # load vectorstores
    # to load need to define which embeddings were used:
    model_name = 'text-embedding-ada-002'
    embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai.api_key)

    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore




def chat_with_user(vectorstore, msg_ = "Thank you. When can I drive again after my operation?"):
    all_chunks_seen = []
    all_pages_seen = []

    opening_message = f"""Hello Mrs Smith, your operation is in 20 days.\n
    Here is your exercise for today: https://www.youtube.com/watch?v=B19AsoXg59c&ab_channel=SWLEOCElectiveOrthopaedicSurgeryInformation \n
    Do you have any questions about your operation?"""

    print(opening_message)

    # output, pages_seen, chunks_seen = chat('Thank you. When can I drive again after my operation?')
    output, pages_seen, chunks_seen = chat(msg_, vectorstore)
    print(output['text'])

    all_chunks_seen.extend(chunks_seen)
    all_pages_seen.extend(pages_seen)
    print(f"\nChunks accessed by patient: {all_chunks_seen}")
    print(f"\nPages accessed by patient: {all_pages_seen}")

    return output['text'], all_chunks_seen, all_pages_seen
    