import os
from dotenv import load_dotenv

load_dotenv()

project = os.getenv("GOOGLE_CLOUD_PROJECT")
region = os.getenv("GOOGLE_CLOUD_REGION")

print(f"Project: {project}")
print(f"Region: {region}")

# Back to ChatVertexAI - correct current model name
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(
    model="gemini-2.5-flash",
    project=project,
    location=region,
    temperature=0.2,
)

response = llm.invoke("Say hello in one sentence.")
print(f"\nGemini response: {response.content}")
print("\n✅ Connection successful!")