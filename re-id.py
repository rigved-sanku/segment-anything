""" This script is re-id script that uses the SAM model to extract embeddings from images and then visualize them using t-SNE"""


import os
import cv2
from PIL import Image
import plotly.express as px


import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap


# Define the SAM class
# This class is used to load the SAM model and extract image embeddings
# The class also has methods to export the model to ONNX format, quantize the model and load the model in onnxruntime 
# Refer : https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb
class SAM():
    def __init__(self, checkpoint='sam_vit_h_4b8939.pth', model_type='vit_h'):
        self.sam  = sam_model_registry[model_type](checkpoint = checkpoint)
        self.onnx_model = None
        
    def export_onnx(self, onnx_model_path):
        onnx_model = SamOnnxModel(self.sam , return_single_mask=True)
        dynamic_axes = {
            "points_coords": {1: "num_points"},
            "points_labels": {1: "num_points"},
        }
        
        embed_dim = self.sam.prompt_encoder.embed_dim
        embed_size = self.sam.prompt_encoder.image_embedding_size
        mask_input_size = [4* x for x in embed_size]
        
        dummy_inputs = {
            "image_embeddings" : torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
            "point_coords" : torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
            "point_labels" : torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
            "mask_input" : torch.randn(1, 1, *mask_input_size, dtype=torch.float),
            "has_mask_input" : torch.tensor([1], dtype=torch.float),
            "orig_im_size" : torch.tensor([1500, 2250], dtype=torch.float),
        }
        
        output_names = ["masks", "iou_predictions", "low_res_masks"]
        
        with torch.no_grad():
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                onnx_model_path,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
                
    def quantize_onnx(self, onnx_model_path, onnx_model_quantized_path):
        quantize_dynamic(
            model_input=onnx_model_path,
            model_output=onnx_model_quantized_path,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        
    def load_onnx_session(self, onnx_model_path):
        ort_session = onnxruntime.InferenceSession(onnx_model_path)
        return ort_session

    def set_device(self, device='cuda'):
        self.sam.to(device=device)

    def get_image_embedding(self, image):
        predictor = SamPredictor(self.sam)
        predictor.set_image(image)
        image_embedding = predictor.get_image_embedding().cpu().numpy()
        return image_embedding



"""
# Evaluate the embeddings of the images in the test_data folder
# The embeddings are extracted using the SAM model

test_data_path = '/home/user/Activity_Monitoring/test_data'
all_items = os.listdir(test_data_path)
#print(type(all_items[0]))
all_items = sorted(all_items)

print('Opening Data Folder ...')
sam = SAM()
sam.set_device()

all_embeddings = {}

for person_folder in all_items:
    #print(type(person_folder))
    person_path = os.path.join(test_data_path, person_folder)
    
    if os.path.isdir(person_path):
        
        print(f'Getting emebeddings of Person Id {person_folder}')
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            image_embedding = sam.get_image_embedding(image)
            
            print(f'EXtracted image embedding from {image_path} with shape {image_embedding.shape}')
            all_embeddings[image_path] = image_embedding
            
np.save('/home/user/Activity_Monitoring/test_data/emebeddings.npy', all_embeddings, allow_pickle=True)
"""

# Load the embeddings and visualize them using t-SNE
# The embeddings are first flattened and then normalized using StandardScaler
all_embeddings = np.load('/home/user/Activity_Monitoring/test_data/emebeddings.npy', allow_pickle=True).item()
flat_embeddings = []
person_ids = []
hover_names = []

for path, embedding in all_embeddings.items():
    #print(f'Opening Image Path: {path} with embedding shape {embedding.shape}')
    embeddings = np.array(embedding)
    flattened_embedding = embeddings.reshape(embeddings.shape[0], -1)
    #print(f'Opening Image Path: {path} with embedding shape {flattened_embedding.shape}')
    flat_embeddings.append(flattened_embedding)
    path_parts = path.split(os.sep)
    person_ids.append(path_parts[-2])
    hover_names.append(path_parts[-1])
    
    
flat_embeddings = np.array(flat_embeddings)
person_ids = np.array(person_ids)
flat_embeddings = np.reshape(flat_embeddings, (flat_embeddings.shape[0], flat_embeddings.shape[2]))
print(f'Flat embedding shape {flat_embeddings.shape}')
print(f'Person Ids {person_ids.shape}')

scaler = StandardScaler()
flat_normal_embeddings = scaler.fit_transform(flat_embeddings)

"""
# Perform PCA on the embeddings

pca = PCA(n_components=50)
reduced_embeddings = pca.fit_transform(flat_normal_embeddings)
print(f'Redcued embedding shape {reduced_embeddings.shape}')
np.save('/home/user/Activity_Monitoring/test_data/pca_emebeddings.npy', reduced_embeddings, allow_pickle=True)
"""

"""
# t-sne hyper paramater tuning (perplexity)
perplexity = np.arange(5, 55, 5)
divergence = []
for i in perplexity:
    model = TSNE(n_components=2, init="pca", perplexity=i)
    reduced = model.fit_transform(reduced_embeddings)
    divergence.append(model.kl_divergence_)
    
fig = px.line(x=perplexity, y=divergence, markers=True)
fig.update_layout(xaxis_title="Perplexity Values", yaxis_title="Divergence")
fig.update_traces(line_color="red", line_width=1)
fig.show()
"""

# Perform t-SNE on the embeddings
# The embeddings are first reduced using PCA
reduced_embeddings = np.load('/home/user/Activity_Monitoring/test_data/pca_emebeddings.npy', allow_pickle=True)
tsne = TSNE(n_components=2, perplexity=45, random_state=42)
tsne_embeddings = tsne.fit_transform(reduced_embeddings)

fig = px.scatter(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], color = person_ids, hover_name = hover_names)
fig.update_layout(
    title="t-SNE visualization of person re-id dataset",
    xaxis_title="First t-SNE",
    yaxis_title="Second t-SNE",
)
fig.show()
print('Done')
    
        