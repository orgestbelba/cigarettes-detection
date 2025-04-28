import pandas as pd
import supervision as sv
import cv2
import os
import time
import google.generativeai as genai
import math
import glob

from roboflow import Roboflow
from datetime import datetime
from io import BytesIO
from PIL import Image

from dotenv import load_dotenv

MIN_WIDTH = 40
MIN_HEIGHT = 40
MIN_ASPECT_RATIO = 0.4
MAX_ASPECT_RATIO = 2.5
GEMINI_MODEL_NAME = "gemini-2.5-pro-exp-03-25"
CROPPED_IMAGE_PREFIX = "cropped_"

image_path = "/Users/orgestbelba/VScodeWorkspace/cigarettes-detection/test_images/cigare5.jpg"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(GEMINI_MODEL_NAME)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/Users/orgestbelba/VScodeWorkspace/cigarettes-detection/output/run_{timestamp}"

def detect_cigarette_packs():
    """
    Detects cigarette packs in the input image using the Roboflow model.
    Returns detections and the annotated image.
    """
    rf = Roboflow(api_key="8SXiXaua5dJ6Ef7FRB7N")
    project = rf.workspace().project("grocery_group_")
    model = project.version(18).model

    result = model.predict(image_path, confidence=30).json()
    for idx, pred in enumerate(result["predictions"]):  
        # Filter only predictions with class "0" (cigarette packs)
        filtered_predictions = [pred for pred in result["predictions"] if pred.get("class") == "0"]
        result["predictions"] = filtered_predictions

    detections = sv.Detections.from_inference(result)

    box_annotator = sv.BoxAnnotator()

    image = cv2.imread(image_path)

    annotated_image = box_annotator.annotate(
        scene=image, detections=detections)

    sv.plot_image(image=annotated_image, size=(16, 16))

    return detections, annotated_image


def crop_and_filter_packs(detections, annotated_image):
    """
    Crops detected bounding boxes from the image and filters them based on size and aspect ratio.
    Saves both accepted and rejected images to the output directory.
    Maps the cropped images positions in the full annotated image for latter processing.
    Returns a dictionary mapping cropped filenames to their bounding box coordinates.
    """
    os.makedirs(output_dir, exist_ok=True)

    rejected_dir = os.path.join(output_dir, "rejected")
    os.makedirs(rejected_dir, exist_ok=True)

    shelf_image = cv2.imread(image_path)

    annotated_image_path = os.path.join(output_dir, f"full_annotated_img_{timestamp}.png")
    cv2.imwrite(annotated_image_path, annotated_image)  # save the full annotated image

    detections.xyxy = sorted(detections.xyxy, key=lambda box: (box[1], box[0]))

    segmented_packs = []
    # Initialize the mapping dictionary
    cropped_file_to_box = {}
    # save cropped images of detected boxes
    for idx, box in enumerate(detections.xyxy):
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_image = shelf_image[y_min:y_max, x_min:x_max]
    
        h, w = cropped_image.shape[:2]
        aspect_ratio = w / h if h > 0 else 0

        if h < MIN_HEIGHT or w < MIN_WIDTH or aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            rejected_path = os.path.join(rejected_dir, f"rejected_{idx + 1}.png")
            cv2.imwrite(rejected_path, cropped_image)
            continue

        segmented_packs.append((cropped_image, (x_min, y_min, x_max, y_max)))
    
        cropped_image_path = os.path.join(output_dir, f"{CROPPED_IMAGE_PREFIX}{idx + 1}.png")
        cv2.imwrite(cropped_image_path, cropped_image)  # save the cropped image
        cropped_filename = f"{CROPPED_IMAGE_PREFIX}{idx + 1}.png"
        cropped_file_to_box[cropped_filename] = (x_min, y_min, x_max, y_max)
        
    return cropped_file_to_box

def recognize_cigarette_packs_gemini(image_directory, batch_size=10):
    """
    Sends batches of cropped images to the Gemini model for brand & type recognition.
    Returns a dictionary mapping image filenames to Gemini's predictions.
    """
    results = {}

    all_image_files = sorted(glob.glob(os.path.join(image_directory, "cropped_*.png")))

    if not all_image_files:
        print("No cropped image files found in directory:", image_directory)
        return {}

    num_batches = math.ceil(len(all_image_files) / batch_size)
    print(f"\n* Found {len(all_image_files)} images. Processing in {num_batches} batches of size {batch_size}.")

    for batch_num in range(num_batches):
        start_batch_idx = batch_num * batch_size
        end_batch_idx = start_batch_idx + batch_size
        batch_image_paths = all_image_files[start_batch_idx:end_batch_idx]

        if not batch_image_paths:
            break

        print(f"\nProcessing Batch {batch_num + 1}/{num_batches} ({len(batch_image_paths)} images)...")

        prompt_text = """For each image provided AFTER this text instruction, identify the **brand** and **type/variety** of the cigarette pack. 
Return the results ONLY as one line per image in this exact format:
Image [N]: Brand: [brand name], Type: [variant/type]

Replace [N] with the image number (1 for the first image in this batch, 2 for the second, etc.).
Do not include multiple brands or types for a single image. 
If you cannot identify the brand or the type, use the word 'Unknown'. Do not add any extra explanations, introductions, or summaries.
Typically, from most of the cigarette brands, you can find the brand name on the pack itself.
About the type/variety, you can use the list below to identify the most common types of cigarettes, some of them having the colors of the pack in the name.

Here is a list of common cigarette brands and types for reference:
```json
{
  "cigarette_brands": {
    "Marlboro": [
      "Marlboro Premium Black", "Gold (2.0 Original)", "Red (FWD)", 
      "Flavor Plus (FWD)", "Touch (4 MG)", "Touch (6 MG)", "Touch SSL (2.0)"
    ],
    "Philip Morris": [
      "Quantum Supreme", "Compac6M", "Compac4", "Infinito"
    ],
    "Merit": [
      "Merit Bianca", "Merit (Orange)"
    ],
    "L&M": [
      "L&M Red Label", "L&M Loft 4MG", "L&M Loft 6MG", 
      "L&M Lounge (6MG)", "L&M Loft XL Blue"
    ],
    "Heets": [
      "Silver Selection", "Yellow Selection", "Amber Selection", 
      "Bronze Selection", "Turquoise Selection", "Dimensions Noor", 
      "Dimensions Yugen"
    ],
    "Fiit": [
      "Regular", "Marine", "Regular Sky"
    ]
  }
}"""

        prompt_parts = [prompt_text]  # Reset prompt_parts for each batch
        current_batch_images_data = {}

        for i, image_path in enumerate(batch_image_paths):
            try:
                print(f"  Loading image: {os.path.basename(image_path)}")
                img = Image.open(image_path)
                prompt_parts.append(img)
                print(f"  Image loaded to prompt successfully: {os.path.basename(image_path)}")
                current_batch_images_data[i + 1] = os.path.basename(image_path)
            except FileNotFoundError:
                print(f"  Error: Image file not found: {image_path}")
                continue
            except Exception as e:
                print(f"  Error loading image {image_path}: {e}")
                continue

        if len(prompt_parts) > 1:
            try:
                print("  Sending request to Gemini API...")
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                response = model.generate_content(
                    prompt_parts,
                    safety_settings=safety_settings,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=2000,
                        temperature=0.1
                    )
                )

                time.sleep(1)

                response_text = response.text.strip()

                response_lines = response_text.split('\n')
                img_counter = 1

                for line in response_lines:
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith(f"Image {img_counter}:"):
                        original_filename = current_batch_images_data.get(img_counter)
                        if original_filename:
                            content_part = line.split(":", 1)[1].strip()
                            results[original_filename] = content_part
                        else:
                            print(f"    Warning: Parsed line for Image {img_counter}, but couldn't map back to a filename.")
                        img_counter += 1
                    else:
                        print(f"    Warning: Unexpected line format in response: {line}")

                if (img_counter - 1) != len(current_batch_images_data):
                    print(f"    Warning: Number of parsed results ({img_counter - 1}) doesn't match number of images sent ({len(current_batch_images_data)})")

            except Exception as e:
                print(f"  Error during Gemini API call or processing for batch {batch_num + 1}: {e}")
                for i in range(len(batch_image_paths)):
                    filename = current_batch_images_data.get(i + 1)
                    if filename and filename not in results:
                        results[filename] = "Error during processing"
                time.sleep(5)

    estimated_total_input_tokens = num_batches * 51000  # 51,000 input tokens per batch
    estimated_total_output_tokens = num_batches * 400   # 400 output tokens per batch

    input_token_cost = (estimated_total_input_tokens / 1000) * 0.000125
    output_token_cost = (estimated_total_output_tokens / 1000) * 0.000375
    total_estimated_cost = input_token_cost + output_token_cost

    print(f"\n--- Estimated Gemini Token Usage and Cost ---")
    print(f"Estimated input tokens: {estimated_total_input_tokens}")
    print(f"Estimated output tokens: {estimated_total_output_tokens}")
    print(f"Estimated input cost: ${input_token_cost:.5f}")
    print(f"Estimated output cost: ${output_token_cost:.5f}")
    print(f"Total estimated cost for this run: ${total_estimated_cost:.5f}")

    print("\nRecognition Completed")
    return results

def generate_results_text(gemini_results):
    """
    Saves Gemini model predictions into a text file in the results subdirectory.
    """
    pd.set_option('display.max_rows', None)

    df = pd.DataFrame(gemini_results.items(), columns=["Image", "Prediction"])

    # Save results to text file in /results subfolder
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "gemini_predictions.txt")
    with open(results_path, "w") as f:
        f.write(df.to_string(index=False))

    print(f"\n* Text results saved to: {results_path}")


def generate_full_image_with_labels(gemini_results, annotated_image, cropped_file_to_box):
    """
    Annotates the full image with predicted brand labels based on Gemini results.
    Saves the annotated image to disk.
    """
    for file_name, prediction in gemini_results.items():
        if file_name not in cropped_file_to_box:
            continue
        x_min, y_min, x_max, y_max = cropped_file_to_box[file_name]
        brand_only = prediction.split(",")[0].replace("Brand:", "").strip()
        label_position = (x_min, y_max + 15)
        cv2.putText(
            annotated_image,
            brand_only,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 0, 255),
            1
        )

    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    labeled_annotated_path = os.path.join(results_dir, f"full_annotated_with_labels_{timestamp}.png")
    cv2.imwrite(labeled_annotated_path, annotated_image)
    print(f"* Annotated image with predictions saved to: {labeled_annotated_path}")

if __name__ == "__main__":
    print("\n--- Starting the Pipeline ---\n")
    detections, annotated_image = detect_cigarette_packs()
    cropped_file_to_box = crop_and_filter_packs(detections, annotated_image)
    gemini_results = recognize_cigarette_packs_gemini(output_dir)
    generate_results_text(gemini_results)
    generate_full_image_with_labels(gemini_results, annotated_image, cropped_file_to_box)
    print("\n--- End of the Pipeline ---")