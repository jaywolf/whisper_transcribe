# *whisper_transcribe*

# Speech-to-Text Transcription with Whisper

This script utilizes OpenAI's Whisper model for automatic speech recognition (ASR) to transcribe audio files into text. The implementation is based on the Hugging Face Transformers library and leverages GPU acceleration if available.

## Setup Instructions

### Prerequisites

- Python 3.8 or later
- `pip` (Python package installer)

### Installation

1. Clone repository to your local machine.
   
   ```bash
   git clone https://github.com/jaywolf/whisper_transcribe.git
   cd whisper_transcribe
   ```

2. Install the required Python packages using `pip` and the provided `requirements.txt` file.
   
   ```bash
   pip install -r requirements.txt
   ```

### Running the Script

1. Ensure you have your audio file named `file_name.m4a` in the project directory.

2. Run the script to transcribe the audio file.
   
   ```bash
   python app.py
   ```

3. The transcription result will be saved in a file named `file_name.txt` in the same directory.

## Code Explanation

The main components of the script are as follows:

- Import necessary libraries and modules:
  
  ```python
  import torch
  from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
  ```

- Set the device to use GPU if available, otherwise fallback to CPU:
  
  ```python
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
  ```

- Define the model ID and load the processor and model from the Hugging Face model hub:
  
  ```python
  model_id = "openai/whisper-large-v3"
  processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
  model = AutoModelForSpeechSeq2Seq.from_pretrained(
      model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
  )
  model.to(device)
  ```

- Create a pipeline for automatic speech recognition:
  
  ```python
  pipe = pipeline(
      "automatic-speech-recognition",
      model=model,
      tokenizer=processor.tokenizer,
      feature_extractor=processor.feature_extractor,
      max_new_tokens=128,
      chunk_length_s=30,
      batch_size=8,
      return_timestamps=True,
      torch_dtype=torch_dtype,
      device=device,
  )
  ```

- Specify the input audio file and output text file, and perform the transcription:
  
  ```python
  content = "file_name"
  result = pipe(content + ".m4a")
  with open(content + ".txt", 'w') as f:
      f.write(str(result["text"]))
  ```

## Acknowledgments

- OpenAI for the Whisper model.
- Hugging Face for the Transformers library.
