import os
import requests
import json
import time
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_audio_uri(audio_file_path):
    """Get the SAS URI for the audio file"""
    # You can either:
    # 1. Set the URI directly in your .env file as AUDIO_URI
    # 2. Or provide it as a parameter
    
    # Check if URI is provided in environment variables
    audio_uri = os.getenv('AUDIO_URI')
    if audio_uri:
        print(f"Using audio URI from environment: {audio_uri}")
        return audio_uri
    
    # If no URI in env, prompt user to provide it
    print(f"Please provide the SAS URI for {audio_file_path}")
    print("You can set it in your .env file as: AUDIO_URI=https://your-storage-account.blob.core.windows.net/container/audio1.mp4?sv=...")
    
    # For now, return a placeholder - you'll need to replace this with your actual SAS URI
    return "https://your-storage-account.blob.core.windows.net/container/audio1.mp4?sv=YOUR_SAS_TOKEN"

def transcribe_audio(audio_file_path):
    """Transcribe audio file using Azure Speech Services Batch API"""
    
    # Get Azure Speech configuration from environment variables
    speech_key = os.getenv('AZURE_SPEECH_KEY')
    speech_endpoint = os.getenv('AZURE_SPEECH_ENDPOINT')
    
    # Validate configuration
    if not speech_key:
        raise ValueError("AZURE_SPEECH_KEY environment variable is not set")
    if not speech_endpoint:
        raise ValueError("AZURE_SPEECH_ENDPOINT environment variable is not set")
    
    # Validate file exists
    if not os.path.exists(audio_file_path):
        raise ValueError(f"Audio file not found: {audio_file_path}")
    
    print(f"Transcribing audio file: {audio_file_path}")
    print(f"Using endpoint: {speech_endpoint}")
    
    # For batch transcription, we need to upload the file first
    # In a real scenario, you'd upload to Azure Blob Storage
    # For this demo, we'll use the direct file approach
    
    try:
        # Get the SAS URI for the audio file
        audio_uri = get_audio_uri(audio_file_path)
        
        # Create transcription request
        transcription_data = {
            "displayName": "Simple transcription",
            "description": "Simple transcription description",
            "locale": "en-US",
            "contentUrls": [audio_uri],
            "properties": {
                "wordLevelTimestampsEnabled": False,
                "punctuationMode": "DictatedAndAutomatic",
                "profanityFilterMode": "Masked"
            }
        }
        
        # Batch transcription API endpoint
        url = f"{speech_endpoint}speechtotext/v3.1/transcriptions"
        
        headers = {
            'Ocp-Apim-Subscription-Key': speech_key,
            'Content-Type': 'application/json'
        }
        
        print("Creating transcription job...")
        response = requests.post(url, headers=headers, json=transcription_data)
        
        if response.status_code == 201:
            # Get transcription ID from location header
            location = response.headers.get('location')
            transcription_id = location.split('/')[-1]
            print(f"Transcription job created with ID: {transcription_id}")
            
            # Poll for completion
            return poll_transcription_status(speech_endpoint, speech_key, transcription_id)
        else:
            print(f"Failed to create transcription: {response.status_code}")
            print(f"Response: {response.text}")
            raise Exception(f"Failed to create transcription: {response.status_code}")
            
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise Exception(f"Transcription failed: {str(e)}")

def poll_transcription_status(endpoint, key, transcription_id):
    """Poll transcription status until completion"""
    url = f"{endpoint}speechtotext/v3.1/transcriptions/{transcription_id}"
    headers = {'Ocp-Apim-Subscription-Key': key}
    
    print("Polling transcription status...")
    
    while True:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            status = result.get('status')
            print(f"Transcription status: {status}")
            
            if status == "Succeeded":
                # Get transcription results
                return get_transcription_results(endpoint, key, transcription_id)
            elif status == "Failed":
                error_msg = result.get('properties', {}).get('error', {}).get('message', 'Unknown error')
                raise Exception(f"Transcription failed: {error_msg}")
            elif status in ["Running", "NotStarted"]:
                print("Transcription in progress, waiting...")
                time.sleep(5)
            else:
                print(f"Unknown status: {status}")
                time.sleep(5)
        else:
            raise Exception(f"Failed to get transcription status: {response.status_code}")

def get_transcription_results(endpoint, key, transcription_id):
    """Get transcription results"""
    url = f"{endpoint}speechtotext/v3.1/transcriptions/{transcription_id}/files"
    headers = {'Ocp-Apim-Subscription-Key': key}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        files = response.json().get('values', [])
        
        for file_data in files:
            if file_data.get('kind') == 'Transcription':
                results_url = file_data.get('links', {}).get('contentUrl')
                if results_url:
                    results_response = requests.get(results_url)
                    if results_response.status_code == 200:
                        results = results_response.json()
                        # Extract the transcription text
                        if 'recognizedPhrases' in results:
                            text_parts = []
                            for phrase in results['recognizedPhrases']:
                                if 'nBest' in phrase and phrase['nBest']:
                                    text_parts.append(phrase['nBest'][0]['display'])
                            return ' '.join(text_parts)
    
    raise Exception("Could not retrieve transcription results")

def summarize_text(text):
    """Summarize text using Azure OpenAI"""
    
    # Get Azure OpenAI configuration from environment variables
    openai_key = os.getenv('AZURE_OPENAI_KEY')
    openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    
    if not openai_key or not openai_endpoint:
        raise ValueError("Azure OpenAI key and endpoint must be set in environment variables")
    
    # Initialize Azure OpenAI client
    client = AzureOpenAI(
        api_key=openai_key,
        api_version="2024-02-15-preview",
        azure_endpoint=openai_endpoint
    )
    
    # Create summarization prompt
    prompt = f"""Please provide a concise summary of the following text in 200 words:

{text}

Summary:"""
    
    # Generate summary
    response = client.chat.completions.create(
        model="gpt-4",  # Change model name as needed
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.3
    )
    
    return response.choices[0].message.content

def main():
    """Main function to transcribe and summarize audio"""
    audio_file = "audio1.mp4"
    
    try:
        print("Starting transcription...")
        transcript = transcribe_audio(audio_file)
        print(f"Transcription completed:\n{transcript}\n")
        
        print("Starting summarization...")
        summary = summarize_text(transcript)
        print(f"Summary:\n{summary}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
