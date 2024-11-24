import os
import cv2
import pytesseract
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Configure pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Update with your path

# Load pre-trained MobileNetV2 model for image classification
model = MobileNetV2(weights='imagenet')


def extract_text_from_image(image_path):
    """Extract text from image using pytesseract"""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {e}"


def identify_objects_in_image(image_path):
    """Identify objects in an image using MobileNetV2 model"""
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (224, 224))  # Resize to MobileNetV2 input size
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Predict
        predictions = model.predict(img_array)
        decoded_preds = decode_predictions(predictions, top=3)[0]

        # Get top 3 predictions
        objects_detected = []
        for _, label, prob in decoded_preds:
            objects_detected.append(f"{label} ({prob * 100:.2f}%)")

        return objects_detected
    except Exception as e:
        return [f"Error identifying objects: {e}"]


def process_images_in_folder(folder_path):
    """Process all images in the folder and return results"""
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            image_path = os.path.join(folder_path, file_name)

            # Extract text from image
            text = extract_text_from_image(image_path)

            # Identify objects in the image
            objects = identify_objects_in_image(image_path)
            objects_str = ', '.join(objects) if objects else "None"

            # If there's no text, print a message in the extracted text field
            if text:
                results.append((file_name, objects_str, text))  # If text exists, add it
            else:
                results.append((file_name, objects_str, "No text in this image"))  # If no text, add message

    return results


def display_results(results):
    """Display results in the UI"""
    # Clear any previous results
    for row in treeview.get_children():
        treeview.delete(row)

    # Insert new results into the treeview
    for file_name, objects, text in results:
        treeview.insert('', 'end', values=(file_name, objects, text))


def on_select_folder():
    """Callback function for folder selection"""
    folder_path = filedialog.askdirectory(title="Select Folder of Images")
    if folder_path:
        try:
            results = process_images_in_folder(folder_path)
            display_results(results)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing the images: {e}")


# Set up the main window (UI)
root = tk.Tk()
root.title("Image Text and Object Detection")

# Make the window resizable
root.geometry("800x600")  # Start with a default size
root.resizable(True, True)  # Allow resizing both horizontally and vertically

# Create a frame for the button and results
frame = tk.Frame(root)
frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Folder selection button (centered)
select_button = tk.Button(frame, text="Select Folder", command=on_select_folder)
select_button.pack(padx=5, pady=5)  # Center button in the frame

# Treeview for displaying results in a table format
columns = ("Image Name", "Detected Objects", "Extracted Text")  # Reversed column order

# Create the Treeview widget
treeview = ttk.Treeview(root, columns=columns, show="headings", height=15)
treeview.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Define column headings and stretch the columns to be resizable
for col in columns:
    treeview.heading(col, text=col, anchor="w")
    treeview.column(col, anchor="w", stretch=tk.YES)  # Allow columns to expand

# Add a horizontal scrollbar for the Treeview
scrollbar = tk.Scrollbar(root, orient="horizontal", command=treeview.xview)
scrollbar.pack(side="bottom", fill="x")

treeview.config(xscrollcommand=scrollbar.set)

root.mainloop()
