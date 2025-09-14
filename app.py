import os
import requests
import time
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def transcribe_container():
    """Transcribe all audio files in Azure Storage container"""
    
    # Get configuration
    speech_key = os.getenv('AZURE_SPEECH_KEY')
    speech_endpoint = os.getenv('AZURE_SPEECH_ENDPOINT')
    container_uri = os.getenv('CONTAINER_URI')
    
    if not all([speech_key, speech_endpoint, container_uri]):
        raise ValueError("Missing required environment variables: AZURE_SPEECH_KEY, AZURE_SPEECH_ENDPOINT, CONTAINER_URI")
    
    print(f"Transcribing container: {container_uri}")
    
    # Create transcription request
    transcription_data = {
        "displayName": "Container transcription",
        "description": "Transcribe all audio files in container",
        "locale": "en-US",
        "contentContainerUrl": container_uri,
        "properties": {
            "wordLevelTimestampsEnabled": False,
            "punctuationMode": "DictatedAndAutomatic",
            "profanityFilterMode": "Masked"
        }
    }
    
    # Submit transcription job
    url = f"{speech_endpoint}speechtotext/v3.1/transcriptions"
    headers = {
        'Ocp-Apim-Subscription-Key': speech_key,
        'Content-Type': 'application/json'
    }
    
    print("Creating transcription job...")
    response = requests.post(url, headers=headers, json=transcription_data)
    
    if response.status_code != 201:
        raise Exception(f"Failed to create transcription: {response.status_code} - {response.text}")
    
    # Get transcription ID
    location = response.headers.get('location')
    transcription_id = location.split('/')[-1]
    print(f"Transcription job created with ID: {transcription_id}")
    
    # Poll for completion
    return poll_transcription(speech_endpoint, speech_key, transcription_id)

def poll_transcription(endpoint, key, transcription_id):
    """Poll transcription status until completion"""
    url = f"{endpoint}speechtotext/v3.1/transcriptions/{transcription_id}"
    headers = {'Ocp-Apim-Subscription-Key': key}
    
    print("Polling transcription status...")
    
    while True:
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Failed to get transcription status: {response.status_code}")
        
        result = response.json()
        status = result.get('status')
        print(f"Transcription status: {status}")
        
        if status == "Succeeded":
            return get_results(endpoint, key, transcription_id)
        elif status == "Failed":
            error_msg = result.get('properties', {}).get('error', {}).get('message', 'Unknown error')
            raise Exception(f"Transcription failed: {error_msg}")
        elif status in ["Running", "NotStarted"]:
            print("Transcription in progress, waiting...")
            time.sleep(5)
        else:
            print(f"Unknown status: {status}")
            time.sleep(5)

def get_results(endpoint, key, transcription_id):
    """Get transcription results"""
    url = f"{endpoint}speechtotext/v3.1/transcriptions/{transcription_id}/files"
    headers = {'Ocp-Apim-Subscription-Key': key}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to get results: {response.status_code}")
    
    files = response.json().get('values', [])
    all_transcriptions = {}
    
    for file_data in files:
        if file_data.get('kind') == 'Transcription':
            filename = file_data.get('name', 'unknown')
            results_url = file_data.get('links', {}).get('contentUrl')
            
            if results_url:
                results_response = requests.get(results_url)
                if results_response.status_code == 200:
                    results = results_response.json()
                    if 'recognizedPhrases' in results:
                        text_parts = []
                        for phrase in results['recognizedPhrases']:
                            if 'nBest' in phrase and phrase['nBest']:
                                text_parts.append(phrase['nBest'][0]['display'])
                        all_transcriptions[filename] = ' '.join(text_parts)
    
    return all_transcriptions

def summarize_text(text):
    """Summarize text using Azure OpenAI"""
    openai_key = os.getenv('AZURE_OPENAI_KEY')
    openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    if not openai_key or not openai_endpoint:
        raise ValueError("Missing Azure OpenAI configuration")
    
    client = AzureOpenAI(
        api_key=openai_key,
        api_version="2024-02-15-preview",
        azure_endpoint=openai_endpoint
    )
    
    prompt = f"""Please provide a concise summary of the following text in 200 words:

{text}

Summary:"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )
    
    return response.choices[0].message.content

def main():
    """Main function"""
    try:
        print("Starting transcription...")
        transcriptions = transcribe_container()
        
        print(f"\nTranscription completed for {len(transcriptions)} files:")
        combined_text = ""
        
        for filename, transcript in transcriptions.items():
            print(f"\n--- {filename} ---")
            print(f"Transcript: {transcript}")
            combined_text += f"\n{filename}: {transcript}"
        
        print("\n" + "="*50)
        print("Starting summarization...")
        summary = summarize_text(combined_text)
        print(f"Summary:\n{summary}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()