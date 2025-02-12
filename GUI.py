# Starting the GUI part from here
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np
import albumentations as A
from scripts.models import MobileNetV2, ResNet, AlexNet, VGG19
import os



THRESHOLD=62859
def decision_function(image, window_size=18, red_multiplier=5.5, green_multiplier=-9.5, blue_multiplier=-0.5, percentile=10):

    image_rgb = image.astype(np.float32)

    redness_score = image_rgb[:, :, 0] * red_multiplier + image_rgb[:, :, 1] * green_multiplier + image_rgb[:, :, 2] * blue_multiplier

    # remove found spots outside the image
    mask = cv2.imread("scripts/mask.png", cv2.IMREAD_GRAYSCALE)

    redness_score = redness_score - np.abs((mask * np.min(redness_score)))

    # Find the top % reddish pixel values
    threshold = np.percentile(redness_score, 100-percentile)
    high_red_indices = np.where(redness_score >= threshold)

    # Initialize variables for the best score and coordinates
    max_score = -np.inf
    best_coords = (0, 0)

    # Calculate window scores only around high redness indices
    h, w = redness_score.shape
    for row, column in zip(*high_red_indices):
        # Define the top-left corner of the window
        top_left_row = max(0, row - window_size // 2)
        top_left_column = max(0, column - window_size // 2)

        # Define the bottom-right corner of the window
        bottom_right_row = min(h, top_left_row + window_size)
        bottom_right_column = min(w, top_left_column + window_size)

        # Extract the window and calculate its redness score
        window = redness_score[top_left_row:bottom_right_row, top_left_column:bottom_right_column]
        score = np.sum(window)

        if score > max_score:
            max_score = score
            best_coords = (row, column)
    
    # Highlight the most reddish area on the image
    image_rgb = image_rgb.astype(np.uint8)
    
    left_upper_column_row = [max(0, best_coords[1] - window_size//2), max(0, best_coords[0] - window_size//2)]
    right_lower_column_row = [min(w, best_coords[1] + window_size//2), min(h, best_coords[0] + window_size//2)]
    cv2.rectangle(image_rgb, left_upper_column_row, right_lower_column_row, (0, 255, 0), 2)

    return max_score, image_rgb

def is_rgb(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Check the number of channels
    if len(image.shape) == 3 and image.shape[2] == 3:
        return True  # RGB image
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        return False  # Grayscale image
    else:
        raise ValueError("Unsupported image format")

models = {
        "MobileNetV2_no_augmentation_CosineAnnealingLR": MobileNetV2,
        "good_model_mobilenetv2": MobileNetV2,
        # "AlexNet_augementation_StepLR": AlexNet,
        # "MobileNetV2_no_augmentation_CosineAnnealingLR_batch_size16": MobileNetV2,
        "MobileNetV2_no_augmentation_StepLR": MobileNetV2,
        # "ResNet_augementation_StepLR": ResNet,
        # "ResNet_no_augmentation_CosineAnnlealingLR_batch_size_16": ResNet,
        # "ResNet_no_augmentation_StepLR": ResNet,
        # "VGG19_augmentation_StepLR": VGG19
    }


# Create a dictionary to map the model names to their corresponding classes
def load_models(model_folder="Models_state_dict"):
    loaded_models = {}

    # Loop through each model and load the corresponding state_dict
    for model_name, model_class in models.items():
        # Construct the path to the saved state_dict file
        state_dict_path = os.path.join(model_folder, f"{model_name}.pth")

        # Load the state_dict from the file
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
            
            # Initialize the model
            model = model_class()
            
            # Load the state_dict into the model
            model.load_state_dict(state_dict)
            
            # Set the model to evaluation mode
            model.eval()
            
            # Store the model in the loaded_models dictionary
            loaded_models[model_name] = model
            print(f"Loaded model: {model_name}")
        else:
            print(f"Model file not found: {state_dict_path}")

    return loaded_models

# Example usage:
loaded_models = load_models()
# Model Mapping
model_mapping = {
    "MobileNet with Cosine Annealing": loaded_models["MobileNetV2_no_augmentation_CosineAnnealingLR"],
    "Mobile NetV2": loaded_models["good_model_mobilenetv2"],
    # "AlexNet": loaded_models["AlexNet_augementation_StepLR"],
    # "MobileNetV2_no_augmentation_CosineAnnealingLR_batch_size16": loaded_models["MobileNetV2_no_augmentation_CosineAnnealingLR_batch_size16"],
    "MobileNet with Step_LR": loaded_models["MobileNetV2_no_augmentation_StepLR"],
    # "ResNet": loaded_models["ResNet_augementation_StepLR"],
    # "ResNet_no_augmentation_CosineAnnlealingLR_batch_size_16": loaded_models["ResNet_no_augmentation_CosineAnnlealingLR_batch_size_16"],
    # "ResNet_no_augmentation_StepLR": loaded_models["ResNet_no_augmentation_StepLR"],
    # "VGG": loaded_models["VGG19_augmentation_StepLR"]
}



def preprocess_image(image_path, mode="RGB"):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if mode.lower() == "gray" else cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Error loading image.")
    
    image = image[32:544, 32:544]
    image[:48, :48] = 0
    image[:31, 452:] = 0
    # augmented = augmentation(image=image)["image"]
    augmented=image

    if augmented.ndim == 3:
        augmented = np.transpose(augmented, (2, 0, 1))
    else:
        augmented = augmented[np.newaxis, ...]
    
    augmented = augmented[np.newaxis, ...]
    return torch.from_numpy(augmented).float()

def load_model(model_name, device):
    model = model_mapping.get(model_name)
    if model:
        model.to(device)
        model.eval()
        return model
    return None

def predict_image(image_path, model, device, mode="RGB"):
    image_tensor = preprocess_image(image_path, mode)
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        return "Healthy" if logits <= 0.5 else "Bleeding"

def open_file_dialog():
    filepath = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if filepath:
        image_path_var.set(filepath)
        display_image(filepath)

def display_image(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((300, 300))  
        image_tk = ImageTk.PhotoImage(image)
        
        image_label.config(image=image_tk)
        image_label.image = image_tk  # Prevent garbage collection
    except Exception as e:
        messagebox.showerror("Image Error", f"Error displaying image: {e}")

def select_model():
    selected_model = model_combobox.get()
    if selected_model == "":
        messagebox.showerror("Error", "Please select a model.")
        return None
    return selected_model

def on_predict_click():
    image_path = image_path_var.get()
    model_name = select_model()
    if not image_path or not model_name:
        messagebox.showerror("Error", "Please provide both image and model.")
        return

    try:
        device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
        model = load_model(model_name, device)
        if model is None:
            messagebox.showerror("Error", "Model loading failed.")
            return
        if is_rgb(image_path):
            mode = "RGB"
        else:
            mode = "gray"  # Grayscale
        
        result = predict_image(image_path, model, device, mode)
        result_label.config(text=f"Prediction: {result}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Tkinter Window
window = tk.Tk()
window.title("Bleeding Detection")
window.geometry("700x400")  

# Configure grid layout
window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=1)

# Left Side - Input Panel
frame_left = tk.Frame(window)
frame_left.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

# Right Side - Image Display
frame_right = tk.Frame(window)
frame_right.grid(row=0, column=1, padx=10, pady=10, sticky="ne")

# Load a Default Blank Image on Startup
def load_default_image():
    blank_image = Image.new("RGB", (300, 300), (400, 400, 400))  # Gray color
    blank_image_tk = ImageTk.PhotoImage(blank_image)
    
    image_label.config(image=blank_image_tk)
    image_label.image = blank_image_tk  # Prevent garbage collection

# Image Display (Top Right) with Default Image
image_label = tk.Label(frame_right, width=300, height=300, bg="gray")
image_label.pack()
load_default_image()  # Load the blank image at startup

# File selection
image_path_var = tk.StringVar()
tk.Label(frame_left, text="Select an Image:").pack(anchor="w")
image_entry = tk.Entry(frame_left, textvariable=image_path_var, width=40)
image_entry.pack(anchor="w")
tk.Button(frame_left, text="Browse", command=open_file_dialog).pack(anchor="w", pady=5)

# Model selection dropdown
tk.Label(frame_left, text="Select a Model:").pack(anchor="w")
model_combobox = ttk.Combobox(frame_left, 
                               values=["Mobile NetV2","MobileNet with Cosine Annealing","MobileNet with Step_LR"],
                            #    values=list(models.keys()),
                               state="readonly", 
                               width=40,   # Adjust width here (number of characters)
                               height=10) 
model_combobox.pack(anchor="w")

# Predict button
tk.Button(frame_left, text="Predict", command=on_predict_click).pack(anchor="w", pady=10)

# Result display
result_label = tk.Label(frame_left, text="Prediction: None")
result_label.pack(anchor="w", pady=5)

# 🔥 Force the UI to render correctly without resizing manually
def force_resize():
    window.update_idletasks()
    window.geometry("701x701")  # Change size slightly
    window.geometry("701x702")  # Restore original size


def on_detect_click():
    image_path = image_path_var.get()
    if not image_path:
        messagebox.showerror("Error", "Please select an image first.")
        return

    try:
        if is_rgb(image_path):
            image = cv2.imread(image_path)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image[32:544, 32:544] # cropping image to get rid of the black borders
        image[:48,:48] = [0,0,0] # painting the upper left corner if there is a gray square
        image[:31, 452:] = [0,0,0] # painting the upper right corner if there is white text parts
        image = np.transpose(image, [2,0,1]) # adjust the axises into the pytorch dimensions of [B, C, W, H]
        image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        score, annotated_image = decision_function(image, *[18, 5.5, -9.5, -0.5, 10])
        print("DOne")

        # Convert to PIL Image
        processed_image_pil = Image.fromarray(annotated_image)
        processed_image_pil = processed_image_pil.resize((300, 300))  # Resize to fit UI
        
        # Convert to Tkinter format and display
        processed_image_tk = ImageTk.PhotoImage(processed_image_pil)
        processed_image_label.config(image=processed_image_tk)
        processed_image_label.image = processed_image_tk  # Prevent garbage collection

                # Save the processed image
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"),
                                                            ("JPEG files", "*.jpg"),
                                                            ("All Files", "*.*")])
        if save_path:
            processed_image_pil.save(save_path)
            messagebox.showinfo("Success", f"Image saved successfully at:\n{save_path}")


                # Update result label based on threshold
        if score < THRESHOLD:
            result_text = f"Redness Score: {score:.2f}, HEALTHY"
            result_color = "green"
        else:
            result_text = f"Redness Score: {score:.2f}, BLEEDING"
            result_color = "red"

        # Update result label
        detection_result_label.config(text=result_text, fg=result_color)


    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Button below the original image
detect_button = tk.Button(frame_right, text="Detect Red Area", command=on_detect_click)
detect_button.pack(pady=5)

# Label for processed image
processed_image_label = tk.Label(frame_right, width=300, height=300, bg="gray")
processed_image_label.pack(pady=5)

# Label to display redness score result
detection_result_label = tk.Label(frame_right, text="Redness Score: N/A", font=("Arial", 12, "bold"))
detection_result_label.pack(pady=5)

# Load a Default Blank Image for Processed Image
def load_default_processed_image():
    blank_image = Image.new("RGB", (300, 300), (200, 200, 200))  # Light gray color
    blank_image_tk = ImageTk.PhotoImage(blank_image)
    
    processed_image_label.config(image=blank_image_tk)
    processed_image_label.image = blank_image_tk  # Prevent garbage collection

# Load the blank image for processed image label at startup
load_default_processed_image()



window.after(100, force_resize)  # Apply fix after 100ms

window.mainloop()
