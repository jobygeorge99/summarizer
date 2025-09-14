import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Get the values directly
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
key = os.getenv("AZURE_OPENAI_KEY")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4")

print(f"Endpoint: {endpoint}")
print(f"Key length: {len(key) if key else 'None'}")
print(f"Deployment: {deployment}")

# Try to create client with minimal parameters
try:
    print("Attempting to create Azure OpenAI client...")
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version="2023-12-01-preview"
    )
    print("✅ Client created successfully!")
    
    # Test a simple completion
    print("Testing chat completion...")
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=50
    )
    
    print("✅ Chat completion successful!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"Error type: {type(e).__name__}")
