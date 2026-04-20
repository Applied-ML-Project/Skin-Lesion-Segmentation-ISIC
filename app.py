import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
import torchvision.transforms as transforms

# Import your model architectures (you'll need to upload attention_unet.py and transunet.py to HF)
from attention_unet import AttentionUNet
from transunet import TransUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize models
att_unet = AttentionUNet().to(device)
trans_unet = TransUNet().to(device)

# Load weights (you'll upload your trained best weights from Kaggle to HF Spaces)
try:
    att_unet.load_state_dict(torch.load("AttentionUNet_best.pth", map_location=device))
    trans_unet.load_state_dict(torch.load("TransUNet_best.pth", map_location=device))
except Exception as e:
    print(f"Weights not found locally. Please ensure .pth files are uploaded. Error: {e}")

att_unet.eval()
trans_unet.eval()

# Preprocessing transforms (Must match exactly what was used in training)
transform = A.Compose([
    A.Resize(256, 256)
])

# ImageNet Normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def predict(image, model_choice):
    if image is None:
        return None
    
    # Preprocess
    img_np = np.array(image)
    transformed = transform(image=img_np)
    img_resized = transformed["image"]
    
    # Normalize 
    img_norm = (img_resized / 255.0 - mean) / std
    img_tensor = torch.tensor(img_norm).float().permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        if model_choice == "Attention U-Net":
            output = att_unet(img_tensor)
        else:
            output = trans_unet(img_tensor)
            
        pred_mask = (output > 0.5).float().squeeze().cpu().numpy()
        
    # Create overlay
    pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
    
    # Colorize the mask (Red for Lesion)
    colored_mask = np.zeros_like(img_resized)
    colored_mask[:, :, 0] = pred_mask_uint8 # Red channel
    
    # Blend overlay (alpha = 0.5)
    overlay = cv2.addWeighted(img_resized, 0.6, colored_mask, 0.4, 0)
    
    return overlay

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Skin Lesion Segmentation Assistant") as demo:
    gr.Markdown("# 🔬 Interactive Skin Lesion Segmentation (ISIC 2018)")
    gr.Markdown("Upload a dermoscopic image to get an AI-generated boundary prediction. This tool provides a **Clinical Decision Support System** to assist dermatologists by highlighting potential lesion boundaries.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Dermoscopy Image")
            model_selector = gr.Radio(["Attention U-Net", "TransUNet"], label="Select Neural Architecture", value="TransUNet")
            submit_btn = gr.Button("Analyze Lesion", variant="primary")
            
        with gr.Column():
            output_overlay = gr.Image(label="AI Segmentation Overlay")
            
            with gr.Accordion("Clinical Interpretation Note", open=False):
                gr.Markdown("""
                - **Attention U-Net**: Generally provides tighter, more localized boundary adherence.
                - **TransUNet**: Provides highly consistent global shapes, robust against visual artifacts like hair or air bubbles.
                
                *Disclaimer: This is an educational tool. All predictions must be verified by a board-certified dermatologist.*
                """)
                
    submit_btn.click(
        fn=predict,
        inputs=[input_image, model_selector],
        outputs=[output_overlay]
    )
    
    # Provide examples if you have some sample images in the repo
    # gr.Examples(["sample1.jpg", "sample2.jpg"], inputs=input_image)

if __name__ == "__main__":
    demo.launch()
