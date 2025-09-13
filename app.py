import time
import requests
import os
import openai
import logging
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AudioFile:
    id: str
    url: str

@dataclass
class Config:
    azure_speech_key: str
    azure_speech_region: str
    azure_openai_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str
    third_party_api: str
    run_interval: int = 300
    poll_interval: int = 10
    max_poll_attempts: int = 60
    audio_uri: Optional[str] = None

def load_config() -> Config:
    """Load and validate configuration from environment variables."""
    required_vars = {
        "AZURE_SPEECH_KEY": os.getenv("AZURE_SPEECH_KEY"),
        "AZURE_SPEECH_REGION": os.getenv("AZURE_SPEECH_REGION"),
        "AZURE_OPENAI_KEY": os.getenv("AZURE_OPENAI_KEY"),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return Config(
        azure_speech_key=required_vars["AZURE_SPEECH_KEY"],
        azure_speech_region=required_vars["AZURE_SPEECH_REGION"],
        azure_openai_key=required_vars["AZURE_OPENAI_KEY"],
        azure_openai_endpoint=required_vars["AZURE_OPENAI_ENDPOINT"],
        azure_openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        third_party_api=os.getenv("THIRD_PARTY_API", "http://localhost:5000/conversation"),
        run_interval=int(os.getenv("RUN_INTERVAL", "300")),
        poll_interval=int(os.getenv("POLL_INTERVAL", "10")),
        max_poll_attempts=int(os.getenv("MAX_POLL_ATTEMPTS", "60")),
        audio_uri=os.getenv("AUDIO_URI", "http://localhost:5000/audio")
    )



def fetch_json_texts(config: Config) -> List[Dict]:
    """Fetch texts from third-party API with error handling."""
    try:
        logger.info(f"Fetching texts from: {config.third_party_api}")
        resp = requests.get(config.third_party_api, timeout=30)
        resp.raise_for_status()
        
        # Check if response is JSON or plain text
        content_type = resp.headers.get('content-type', '').lower()
        
        if 'application/json' in content_type:
            # Handle JSON response
            data = resp.json()
            if not isinstance(data, list):
                raise ValueError("Expected list response from API")
            
            texts = []
            for item in data:
                if isinstance(item, dict) and "id" in item and "text" in item:
                    texts.append(item)
                else:
                    logger.warning(f"Skipping invalid item: {item}")
            
            logger.info(f"Found {len(texts)} JSON texts")
            return texts
        else:
            # Handle plain text response (like from conversation API)
            text_content = resp.text.strip()
            if text_content:
                # Create a single text item from the plain text response
                text_item = {
                    "id": f"transcript_{int(time.time())}",
                    "text": text_content
                }
                logger.info(f"Found 1 text transcript ({len(text_content)} characters)")
                return [text_item]
            else:
                logger.warning("Empty response from API")
                return []
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch texts: {e}")
        return []
    except (ValueError, KeyError) as e:
        logger.error(f"Invalid response format: {e}")
        return []

def start_transcription(config: Config, audio_url: str) -> str:
    """Submit audio to Azure Speech v3.2 batch transcription and return job URL."""
    try:
        url = f"https://{config.azure_speech_region}.api.cognitive.microsoft.com/speechtotext/v3.2/transcriptions"
        headers = {
            "Ocp-Apim-Subscription-Key": config.azure_speech_key,
            "Content-Type": "application/json"
        }
        body = {
            "contentUrls": [audio_url],
            "locale": "en-US",
            "displayName": f"Transcription_{int(time.time())}"
        }
        
        logger.info(f"Starting transcription for: {audio_url}")
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        
        job_url = resp.headers.get("Location")
        if not job_url:
            raise ValueError("No job URL returned from Azure Speech")
        
        logger.info(f"Transcription job started: {job_url}")
        return job_url
        
    except requests.RequestException as e:
        logger.error(f"Failed to start transcription: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid response: {e}")
        raise

def poll_transcription(config: Config, job_url: str) -> str:
    """Poll Azure Speech job until completion with timeout; return transcript URL."""
    headers = {"Ocp-Apim-Subscription-Key": config.azure_speech_key}
    attempts = 0
    
    while attempts < config.max_poll_attempts:
        try:
            resp = requests.get(job_url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            status = data.get("status")
            logger.info(f"Job status: {status}")
            
            if status == "Succeeded":
                transcript_url = data.get("resultsUrls", {}).get("transcription")
                if transcript_url:
                    logger.info("Transcription completed successfully")
                    return transcript_url
                else:
                    raise ValueError("No transcript URL in successful response")
            elif status == "Failed":
                error_msg = data.get("error", {}).get("message", "Unknown error")
                raise Exception(f"Transcription failed: {error_msg}")
            elif status in ["NotStarted", "Running"]:
                logger.info(f"Job still running, attempt {attempts + 1}/{config.max_poll_attempts}")
            else:
                logger.warning(f"Unknown status: {status}")
            
            attempts += 1
            time.sleep(config.poll_interval)
            
        except requests.RequestException as e:
            logger.error(f"Failed to poll job status: {e}")
            attempts += 1
            time.sleep(config.poll_interval)
    
    raise TimeoutError(f"Transcription job timed out after {config.max_poll_attempts} attempts")

def get_transcript(transcript_url: str) -> str:
    """Fetch transcript text from Azure Speech service."""
    try:
        logger.info(f"Fetching transcript from: {transcript_url}")
        resp = requests.get(transcript_url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch transcript: {e}")
        raise

def summarize(config: Config, text: str) -> str:
    """Summarize JSON text using Azure OpenAI."""
    try:
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            api_key=config.azure_openai_key,
            api_version="2024-06-01",
            azure_endpoint=config.azure_openai_endpoint
        )

        logger.info("Generating summary with Azure OpenAI")
        response = client.chat.completions.create(
            model=config.azure_openai_deployment,
            messages=[
                {"role": "system", "content": "Summarize the following JSON text clearly and concisely, maximum 50 words"},
                {"role": "user", "content": text}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content
        logger.info("Summary generated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        raise

def process_json_text(config: Config, json_item: Dict) -> bool:
    """Process a single JSON text item through the summarization pipeline."""
    try:
        logger.info(f"Processing JSON text: {json_item.get('id', 'unknown')}")
        
        # Get text from JSON item
        text = json_item.get('text', '')
        if not text:
            logger.warning(f"No text found in JSON item: {json_item}")
            return False
        
        # Generate summary
        summary = summarize(config, text)
        
        # Output results
        logger.info(f"--- Original Text (first 300 chars) ---\n{text[:300]}...")
        logger.info(f"--- Summary ---\n{summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process JSON text {json_item.get('id', 'unknown')}: {e}")
        return False

def main_loop(config: Config):
    """Main orchestration loop with error handling."""
    logger.info("Starting JSON text summarization service")
    
    try:
        if True:
            logger.info("Checking for new JSON texts...")
            
            # Fetch JSON texts
            json_texts = fetch_json_texts(config)
            
            if json_texts:
                logger.info(f"Processing {len(json_texts)} JSON texts")
                success_count = 0
                
                for json_item in json_texts:
                    if process_json_text(config, json_item):
                        success_count += 1
                
                logger.info(f"Completed processing: {success_count}/{len(json_texts)} texts successful")
            else:
                logger.info("No JSON texts found")
            
            logger.info(f"Sleeping for {config.run_interval} seconds...")
            time.sleep(config.run_interval)
            
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        raise

def main():
    """Main entry point with configuration loading and validation."""
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
        main_loop(config)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
