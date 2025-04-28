# Cigarette Pack Detection and Recognition

This project detects cigarette packs from images, crops the detected regions, and uses Google Gemini AI to recognize the brand and type of each cigarette pack.

## Features
- Detects cigarette packs in images using a Roboflow-trained model (grocery_group_ v18).
- Filters and crops detected packs based on size and aspect ratio constraints.
- Sends cropped images to Google Gemini AI for brand and type recognition.
- Annotates the original image with the recognized brand names.
- Saves detection results and annotated images for review.

## Requirements
- Python 3.x
- Required Python packages:
  - pandas
  - supervision
  - opencv-python
  - google-generativeai
  - roboflow
  - pillow
  - python-dotenv

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/cigarettes-detection.git
   cd cigarettes-detection
   ```

2. Install dependencies using uv:
   ```
   uv syncs
   ```

3. Set up your `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Configuration

- Update `image_path` in `main.py` to point to your test image.
- Ensure Roboflow API key and project details are correct in `detect_cigarette_packs()`.

## Usage

Run the script:
```
python main.py
```

The script will:
1. Detect and crop cigarette packs from the image.
2. Recognize brands and types using Gemini AI.
3. Save results and annotated images in a timestamped output folder.

## Output

- Cropped images of detected packs.
- Text file listing recognized brands and types.
- Annotated image with detected packs and brand labels.

## License
This project is licensed under the MIT License.