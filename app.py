from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. Initialize the model
llm = ChatOllama(model="llama3.2", temperature=0.7)

# 2. Create a "Template"
# This tells the AI: "You are a social media expert. I will give you a topic, you write the tweet."
prompt = ChatPromptTemplate.from_template("Write a funny and engaging tweet about {topic}.")

# 3. Create a Chain (Connect the Prompt to the Model)
# This is the "LangChain" magic: Prompt | Model
chain = prompt | llm

# 4. Get User Input
user_topic = input("Enter a topic for the tweet: ")

# 5. Run the Chain
print(f"Generating tweet about {user_topic}...")
response = chain.invoke({"topic": user_topic})

print("\nAI Tweet:")
print(response.content)