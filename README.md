# NeuTTS GUI Application

A graphical user interface for the [NeuTTS-nano](https://github.com/neuphonic/neutts) text-to-speech model that allows you to generate speech from text using neural voice cloning.

## Features

- **Text-to-Speech Generation**: Convert text to natural-sounding speech
- **Voice Cloning**: Use reference samples to clone voices
- **Multi-language Support**: English, Spanish, French, ... language models
- **URL Content Extraction**: Extract text from web pages
- **File Reading**: Read text from PDF, TXT, and HTML files
- **Real-time Playback**: Listen to generated audio as it's created
- **Audio Saving**: Save generated audio to WAV files
- **Chunked Processing**: Intelligent text chunking for better audio quality

## Installation

1. Download and install NeuTTS: 
    - [NeuTTS](https://github.com/neuphonic/neutts?tab=readme-ov-file#get-started-with-neutts):
    - [NeuCodec codec model](https://huggingface.co/collections/neuphonic/neucodec)

Tips: 
    - Don’t forget the big files, git clone will not get them (e.g. model.safetensors, tokenizer.json, tokenizer_config.json) 
    - NeuCodec don’t run 100% locally, it needs network each time you load a model (e.g. at start or then switching language), once loaded it will work offline.
    
```bash
git clone https://github.com/neuphonic/neutts
cd neutts/
# Then follow official installation procedure
```

2. Clone this repository:
```bash
git clone https://github.com/ryo-ohki-code/neutts-gui
cd neutts-gui/
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the desired models (English, French, Spanish, German):
    - [NeuTTS backbone models](https://huggingface.co/collections/neuphonic/neutts-nano-multilingual-collection)
    - If needed adjust file paths in the code to match their location (backbone_repo_*, codec_repo_path, ../neutts/samples/)
    - Ensure all model files are accessible

```
.
├── neutts-gui/
├── neutts/                 # From Github neuphonic
├── neucodec/               # From HuggingFace neuphonic
├── neutts-nano/            # Model English version
├── neutts-nano-spanish/    # Download as you need
├── neutts-nano-german/     # Download as you need
└── neutts-nano-french/     # Download as you need
```

## Usage

1. Run the application:
```bash
python neuTTS_GUI.py
```

2. **Input Text**: Enter or paste text in the main text area

3. **Select Reference Sample**: Choose from available voice samples in the `samples` directory

4. **Language Selection**: Choose between English and French

5. **Compute Type**: Select CUDA for GPU acceleration or CPU

6. **Save Audio**: Check "💾 Save audio to file" to save generated audio

7. **Generate Audio**: Click "🔊 Generate & Play" to create speech


## Reference Samples

Place your reference voice samples in the `samples` directory as `.wav` and `.txt` pairs:
- `sample1.wav` - Reference audio file
- `sample1.txt` - Corresponding text for voice cloning

## Troubleshooting

### Common Issues

1. **Missing Models**: Ensure all required model files are downloaded and placed in the correct directories
2. **CUDA Errors**: Verify CUDA installation and GPU compatibility
3. **File Not Found**: Check that reference sample files exist in the samples directory
4. **Audio Playback Issues**: Install PortAudio or check audio device permissions

### Error Messages

- `Reference audio not found`: Check sample file paths
- `Model warm-up skipped`: May occur if model loading fails
- `Network error`: Check internet connection for URL extraction
- `No readable content found`: Web page may be JavaScript-heavy or blocked

## License

MIT License

## Acknowledgments

- NeuTTS: Neural Text-to-Speech system
- NeuCodec: Audio codec for neural speech synthesis
- trafilatura: Web content extraction library
- PyPDF2: PDF file processing library
- SoundDevice: Audio playback library
