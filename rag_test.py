from langchain_community.document_loaders import PyPDFLoader

# 1. Load the PDF
# Make sure your file is named 'data.pdf' and is in the same folder!
loader = PyPDFLoader("DL Concepts.pdf")
docs = loader.load()

# 2. Check if it worked
print(f"Successfully loaded {len(docs)} pages.")
print("Here is a snippet from page 1:")
print(docs[0].page_content[:500]) # Prints first 500 characters