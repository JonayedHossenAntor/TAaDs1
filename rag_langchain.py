from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain_huggingface.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
import sqlite3
from langchain.schema import Document

# Define SQLite database file path and table name
sqlite_file = "./courses.sqlite" 
table_name = "zqm_module_en"
content_columns = ["title", "instructor", "learning_obj", "course_contents", "prerequisites", "readings", "applicability", "workload", "credits", "evaluation", "time", "frequency", "duration", "course_type", "remarks"]

conn = sqlite3.connect(sqlite_file)
cursor = conn.cursor()

# Fetch data from the specified columns
query = f"SELECT {', '.join(content_columns)} FROM {table_name}"
cursor.execute(query)
rows = cursor.fetchall()

# Combine the columns into a single text field for each row
data = []
for row in rows:
    combined_text = " ".join(str(col) for col in row if col is not None)  # Handle None values
    metadata = {content_columns[i]: row[i] for i in range(len(content_columns))}
    data.append({"page_content": combined_text, "metadata": metadata})

# Close the SQLite connection
conn.close()

# Create an instance of RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

docs = [Document(page_content=item['page_content'], metadata=item['metadata']) for item in data]

# Split the text into documents
split_docs = text_splitter.split_documents(docs)


# Define the path to the pre-trained model you want to use
modelPath = "sentence-transformers/all-MiniLM-l6-v2"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

db = FAISS.from_documents(split_docs, embeddings)

# Create a question-answering model object by loading the pretrained "Intel/dynamic_tinybert" model.
model_name = "Intel/dynamic_tinybert"

# Load the tokenizer associated with the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Define a question-answering pipeline using the model and tokenizer
question_answerer = pipeline(
    "question-answering", 
    model=model_name, 
    tokenizer=tokenizer,
    return_tensors='pt'
)

# Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
# with additional model-specific arguments (temperature and max_length)
llm = HuggingFacePipeline(
    pipeline=question_answerer,
    model_kwargs={"temperature": 0.8, "max_length": 512},
)

# Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
retriever = db.as_retriever(search_kwargs={"k": 4})

# Create a question-answering instance (qa) using the RetrievalQA class.
# It's configured with a language model (llm), a chain type "refine," the retriever we created, and an option to not return source documents.
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)

question = "Which courses are taken by Dr Michael Dornieden?"
result = qa.run({"query": question})
print(result["result"])