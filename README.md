```markdown
# Vision-Language Models: Image & Video Understanding

A practical implementation of vision-language models for semantic image retrieval and video question answering. This project demonstrates how modern VLMs can understand and reason about visual content through natural language.

## Overview

This repository contains two core modules:

**Image Understanding** - Implements semantic image search using vision-language embeddings. Images are encoded into a shared embedding space with text, enabling natural language queries to retrieve semantically similar images. This approach powers image search engines, visual similarity systems, and multimodal retrieval pipelines.

**Video Understanding** - Implements video question answering by sampling representative frames and processing them with a vision-language model. The system can answer natural language questions about video content, including scene description, action recognition, and event understanding.

## Features

- CLIP-based image embedding and similarity search with FAISS
- Frame-based video processing for efficient question answering
- Interactive video QA interface with natural language queries
- Semantic understanding of scenes, actions, and events
- Clean separation between perception (embeddings) and reasoning (language models)

## Installation

Create a Python 3.11 environment and install dependencies:

```bash
conda create -n vlm_env python=3.11 -y
conda activate vlm_env
pip install --upgrade pip
pip install torch torchvision torchaudio transformers accelerate pillow ffmpeg-python faiss-cpu datasets
```

Install FFmpeg (required for video processing):

```bash
brew install ffmpeg  # macOS
sudo apt-get install ffmpeg  # Ubuntu/Debian
```

## Usage

Run the video question answering demo:

```bash
python src/video/video_qa.py path/to/video.mp4
```

Example interaction:

```
Question> what is happening in the video?
Question> is it daytime or nighttime?
Question> are pedestrians visible?
Question> exit
```

## Technical Approach

The project uses a two-stage architecture. The perception layer extracts semantic visual representations through vision encoders, while the reasoning layer uses language models to interpret and explain visual content. This design reflects production ML systems where specialized models extract structured information and language models provide human-interpretable explanations.

Image retrieval works by encoding both images and text queries into a shared embedding space where semantically similar concepts are positioned close together. To enable efficient search across large image collections, the system uses FAISS (Facebook AI Similarity Search) to index pre-computed image embeddings. This vector indexing allows for sub-second retrieval times even with millions of images, using approximate nearest neighbor search algorithms instead of exhaustive comparison. The index is built once during preprocessing and loaded at runtime, making queries nearly instantaneous.

Video understanding samples representative frames from the video and processes them alongside questions to generate contextual answers. The temporal sampling ensures the model captures the video's progression while maintaining computational efficiency.

## Strengths and Limitations

This implementation works at semantic understanding tasks including scene description, action recognition, event detection, and high-level visual reasoning. It can answer questions about what is happening in a scene, identify objects and activities, and understand context.

The system is not designed for precise measurements, exact counting, pixel-level analysis, or frame-accurate detection. Vision-language models reason semantically rather than metrically. For tasks requiring precise detection, counting, or tracking, traditional computer vision models are more appropriate.

## Requirements

- Python 3.11
- PyTorch with torchvision and torchaudio
- Hugging Face Transformers and Accelerate
- FFmpeg for video frame extraction
- FAISS for similarity search
- PIL for image processing

## Project Structure

```
src/
├── image/    # Image embedding and retrieval
└── video/    # Video question answering
```

Data files, embeddings, videos, and search results are excluded from version control to keep the repository clean and lightweight.


## Demo samples 

<img width="500" height="700" alt="image" src="https://github.com/user-attachments/assets/e9f02570-ed42-4bd1-b9b0-fb780404a30f" />

This is just with dataset sample of size 500 images from BDD100K dataset

<img width="500" height="700" alt="image" src="https://github.com/user-attachments/assets/92acbd07-f558-4ded-a7e7-11b8af922a95" />

I have uploaded a sample dashcam video from the youtube for this example



