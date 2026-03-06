# 🏗️ Smart Tile AI: Model Architecture

## 📋 Table of Contents
* [1. Room Segmentation (SegFormer)](#1-room-segmentation-the-eyes)
* [2. Visual Search & Matching (CLIP)](#2-visual-search--matching-the-brain)
* [3. Background Extraction (U2-Net)](#3-background-extraction-the-cleaner)

---

### **1. Room Segmentation (The "Eyes")**
* **Model ID**: `nvidia/segformer-b0-finetuned-ade-512-512`
* **Library**: `transformers`
* **Purpose**: Identifies "Floor" and "Wall" pixels to create high-precision masks for tile overlays.
* **Dataset**: Fine-tuned on **ADE20K** for indoor scenes.

### **2. Visual Search & Matching (The "Brain")**
* **Model ID**: `openai/clip-vit-base-patch32`
* **Library**: `transformers`
* **Purpose**: Converts images into mathematical embeddings to compare the user's query against the `catalog_tiles`.

### **3. Background Extraction (The "Cleaner")**
* **Model ID**: `U2-Net`
* **Library**: `rembg`
* **Purpose**: Strips backgrounds from tile photos to ensure a clean texture application.



## Internal Note: These models are automatically downloaded from Hugging Face on the first run. Ensure the server has at least 4GB of free RAM and a stable internet connection for the initial "Materializing" phase.


## Storage Space: The project will download approximately 1.2GB of weights from the Hugging Face Hub during the first execution.