from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from pathlib import Path
import json
from datetime import datetime
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import faiss
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
EMBEDDINGS_DIR = 'my_embeddings'
IMAGES_BASE_DIR = 'AOF'
MAX_FILE_SIZE = 16 * 1024 * 1024  
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'webp'}

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class ImageSearchAPI:
    def __init__(self, embeddings_dir, images_base_dir):
        self.embeddings_dir = Path(embeddings_dir)
        self.images_base_dir = Path(images_base_dir)
        
        # Load FAISS index
        index_path = self.embeddings_dir / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = self.embeddings_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load index info
        info_path = self.embeddings_dir / "index_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.index_info = json.load(f)
        else:
            self.index_info = {}
        
        # Initialize model
        self.model = models.mobilenet_v3_large(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def extract_embedding(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze()
            
            embedding = features.cpu().numpy()
            
            # L2 normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def get_relative_path(self, absolute_path):
        abs_path = Path(absolute_path)
        try:
            relative = str(abs_path.relative_to(self.images_base_dir))
            return relative.replace('\\', '/')
        except ValueError:
            return abs_path.name
    
    def search(self, query_image_path, top_k=10):
        try:
            query_embedding = self.extract_embedding(query_image_path)
            if query_embedding is None:
                return {
                    "success": False,
                    "error": "Failed to extract embedding from query image",
                    "results": [],
                    "total_results": 0
                }
            
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            distances, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx < len(self.metadata):
                    metadata = self.metadata[idx]
                    # Get relative path and normalize to forward slashes
                    relative_path = self.get_relative_path(metadata.get("image_path", ""))
                    
                    # Use forward slashes consistently
                    path_obj = Path(relative_path.replace('\\', '/'))
                    parent_path = str(path_obj.parent)
                    # Ensure location uses forward slashes
                    location = parent_path.replace('\\', '/') if parent_path != '.' else ""
                    
                    result = {
                        "fullPath": relative_path,
                        "id": str(uuid.uuid4()),
                        "location": location,
                        "name": path_obj.stem,
                        "type": "file"
                    }
                    results.append(result)
            
            return {
                "success": True,
                "results": results,
                "total_results": len(results),
                "query_processed": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "total_results": 0,
                "query_processed": False
            }

try:
    search_api = ImageSearchAPI(EMBEDDINGS_DIR, IMAGES_BASE_DIR)
    print("Image search API initialized successfully!")
except Exception as e:
    print(f"Error initializing search API: {e}")
    search_api = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "api_ready": search_api is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    if search_api is None:
        return jsonify({
            "success": False,
            "error": "Search API not initialized"
        }), 500
    
    return jsonify({
        "success": True,
        "stats": {
            "total_images": len(search_api.metadata),
            "index_info": search_api.index_info
        }
    })

@app.route('/api/search', methods=['POST'])
def search_similar_images():
    if search_api is None:
        return jsonify({
            "success": False,
            "error": "Search API not initialized"
        }), 500
    
    # Check if file is in request
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided"
        }), 400
    
    file = request.files['image']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "No file selected"
        }), 400
    
    # Check file type
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    # Check file size
    if request.content_length > MAX_FILE_SIZE:
        return jsonify({
            "success": False,
            "error": f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        }), 400
    
    try:
        # Get parameters
        top_k = int(request.form.get('top_k', 10))
        top_k = min(max(top_k, 1), 50)  
        
        # Save uploaded file
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Perform search
        results = search_api.search(filepath, top_k)
        
        # For Windows
        if results.get('success') and 'results' in results:
            for result in results['results']:
                if 'location' in result:
                    result['location'] = result['location'].replace('\\', '/')
                if 'fullPath' in result:
                    result['fullPath'] = result['fullPath'].replace('\\', '/')
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(results)
        
    except Exception as e:
        # Clean up file if error occurs
        try:
            if 'filepath' in locals():
                os.remove(filepath)
        except:
            pass
        
        return jsonify({
            "success": False,
            "error": f"Search failed: {str(e)}"
        }), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve images from the AOF directory"""
    try:
        return send_from_directory(IMAGES_BASE_DIR, filename)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Image not found: {filename}"
        }), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "success": False,
        "error": f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
    }), 413

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    app.run(debug=True, host='0.0.0.0', port=5000)