import os
import cv2
from PIL import Image
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


""""
test_data_path = '/home/user/Activity_Monitoring/test_data'
print('Opening Data Folder ...')
sam = SAM()
sam.set_device()

all_embeddings = {}

for person_folder in os.listdir(test_data_path):
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
            
np.save('/home/user/Activity_Monitoring/test_data/emebeddings.npy', all_embeddings)
"""

       
all_embeddings = np.load('/home/user/Activity_Monitoring/test_data/emebeddings.npy', allow_pickle=True)

for path, embedding in all_embeddings.item():
    print(f'Opening Image Path: {path} with embedding shape {embedding.shape}')
    embeddings = np.array(embedding)
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1)
    print(f'Opening Image Path: {path} with embedding shape {flattened_embeddings.shape}')
    pca = PCA(n_components=50)
    reduced_emebeddings = pca.fit_transform(flattened_embeddings)
    print(f'Opening Image Path: {path} with embedding shape {reduced_emebeddings.shape}')
        
    
        