import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the values
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
key = os.getenv("AZURE_OPENAI_KEY")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4")

print(f"Endpoint: {endpoint}")
print(f"Key length: {len(key) if key else 'None'}")
print(f"Deployment: {deployment}")

# Try different import and initialization methods
print("\n=== Method 1: Standard AzureOpenAI ===")
try:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version="2023-12-01-preview"
    )
    print("✅ Method 1: Client created successfully!")
    
    response = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=50
    )
    print(f"✅ Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ Method 1 failed: {e}")

print("\n=== Method 2: OpenAI with Azure endpoint ===")
try:
    import openai
    openai.api_type = "azure"
    openai.api_base = endpoint
    openai.api_version = "2023-12-01-preview"
    openai.api_key = key
    
    response = openai.ChatCompletion.create(
        engine=deployment,
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=50
    )
    print(f"✅ Method 2 Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ Method 2 failed: {e}")

print("\n=== Method 3: Direct requests ===")
try:
    import requests
    import json
    
    url = f"{endpoint}openai/deployments/{deployment}/chat/completions?api-version=2023-12-01-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": key
    }
    data = {
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 50
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Method 3 Response: {result['choices'][0]['message']['content']}")
    else:
        print(f"❌ Method 3 failed: {response.status_code} - {response.text}")
        
except Exception as e:
    print(f"❌ Method 3 failed: {e}")
