# From https://github.com/ljwztc/CLIP-Driven-Universal-Model
import os
import clip
import torch


## PAOT
# ORGAN_NAME = ['Background', 'Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus', 
#                 'Liver', 'Stomach', 'Aorta', 'Inferior Vena Cava', 'Portal Vein and Splenic Vein',
#                 'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland']

# # Load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)


# text_inputs = torch.cat([clip.tokenize(f'A computerized tomography of a {item}') for item in ORGAN_NAME]).to(device)

# # Calculate text embedding features
# with torch.no_grad():
#     text_features = model.encode_text(text_inputs)
#     print(text_features.shape, text_features.dtype)
#     torch.save(text_features, 'txt_encoding2.pth')
# Define organ names and descriptions
# ORGAN_DESCRIPTIONS = {
#     "Background": "all regions in the CT scan that do not belong to labeled anatomical structures.",
#     "Spleen": "a lymphatic organ involved in blood filtration and immune response, located in the upper left abdomen.",
#     "Right Kidney": "the right-side kidney that filters blood and produces urine; slightly lower due to liver position.",
#     "Left Kidney": "the left-side kidney responsible for filtering blood and producing urine.",
#     "Gall Bladder": "a small organ beneath the liver that stores and releases bile to aid in fat digestion.",
#     "Esophagus": "a muscular tube that transports food and liquids from the mouth to the stomach.",
#     "Liver": "the largest internal organ, responsible for detoxifying blood and producing bile.",
#     "Stomach": "a hollow organ that digests food using acid and enzymes before it enters the intestine.",
#     "Aorta": "the body's main artery that carries oxygen-rich blood from the heart to the rest of the body.",
#     "Inferior Vena Cava": "a large vein that returns deoxygenated blood from the lower body to the heart.",
#     "Portal Vein and Splenic Vein": "vessels that transport nutrient-rich blood from the gastrointestinal organs to the liver.",
#     "Pancreas": "a gland that produces digestive enzymes and hormones like insulin for blood sugar regulation.",
#     "Right Adrenal Gland": "a hormone-producing gland on top of the right kidney, responsible for adrenaline and cortisol.",
#     "Left Adrenal Gland": "a small endocrine gland above the left kidney that produces essential stress hormones."
# }
# ORGAN_DESCRIPTIONS = {
#     "Background": "regions outside the labeled abdominal organs, including surrounding fat, muscle, and air spaces.",
#     "Spleen": "a crescent-shaped organ in the left upper abdomen, posterior to the stomach and lateral to the left kidney.",
#     "Right Kidney": "a bean-shaped organ located in the right mid-to-upper abdomen, posteriorly adjacent to the spine and below the liver.",
#     "Left Kidney": "a bean-shaped organ in the left mid-to-upper abdomen, posterior to the spleen and lateral to the aorta.",
#     "Gall Bladder": "a small, oval structure located beneath the right lobe of the liver, near the anterior abdominal wall.",
#     "Esophagus": "a narrow, vertical tube running along the posterior mediastinum, just anterior to the vertebral column and left of the midline in the upper abdomen.",
#     "Liver": "a large, triangular organ occupying most of the right upper quadrant, extending across the midline in upper slices.",
#     "Stomach": "a J-shaped organ in the left upper abdomen, inferior to the diaphragm and medial to the spleen.",
#     "Aorta": "a round vessel positioned slightly left of the vertebral body, running vertically through the retroperitoneum.",
#     "Inferior Vena Cava": "an elliptical vessel located anterior and to the right of the aorta in the retroperitoneal space.",
#     "Portal Vein and Splenic Vein": "branched vascular structures near the liver hilum and pancreas, with horizontal course across upper abdominal slices.",
#     "Pancreas": "an elongated, lobular organ spanning from the right to left upper abdomen, posterior to the stomach and anterior to the spine.",
#     "Right Adrenal Gland": "a small triangular structure superior to the right kidney, between the kidney and liver, near the posterior abdominal wall.",
#     "Left Adrenal Gland": "a semilunar structure above the left kidney, medial to its upper pole and adjacent to the aorta."
# }

# Generate rich text prompts
text_prompts = [
    f"A computerized tomography of a {name}, which is {desc}"
    for name, desc in ORGAN_DESCRIPTIONS.items()
]

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device=device)

# Tokenize and encode
text_inputs = torch.cat([clip.tokenize(p) for p in text_prompts]).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_inputs)
    print(text_features.shape, text_features.dtype)
    torch.save(text_features, 'txt_encoding_rich2.pth')

