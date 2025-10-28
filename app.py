from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from pathlib import Path
import json
from datetime import datetime
import threading
import logging

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import faiss
import imagehash
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {
    "origins": "*",
    "methods": ["GET", "POST"],
    "allow_headers": ["Content-Type"],
    "expose_headers": ["Cache-Control", "Pragma", "Expires"]
}})

# Configuration
UPLOAD_FOLDER = 'uploads'
EMBEDDINGS_BASE_DIR = 'embeddings'
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_IMAGE_DIMENSION = 10000  # Max width or height in pixels
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tif', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EMBEDDINGS_BASE_DIR, exist_ok=True)

class ImageSearchAPI:
    """Image search API for a single company."""

    def __init__(self, embeddings_dir, images_base_dir, shared_model, shared_transform):
        self.embeddings_dir = Path(embeddings_dir)
        self.images_base_dir = Path(images_base_dir)
        self.model = shared_model
        self.transform = shared_transform

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
    
    def validate_image_dimensions(self, image_path):
        """Validate image dimensions are within acceptable limits."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                    return False, f"Image dimensions too large ({width}x{height}). Max: {MAX_IMAGE_DIMENSION}px"
                return True, None
        except Exception as e:
            return False, f"Error validating image: {str(e)}"

    def extract_embedding(self, image_path):
        try:
            # Validate dimensions
            valid, error = self.validate_image_dimensions(image_path)
            if not valid:
                logger.warning(error)
                # Continue anyway but log the warning

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

    def compute_image_hash(self, image_path):
        """Compute perceptual hash of an image for duplicate detection."""
        try:
            image = Image.open(image_path)
            # Use average hash for duplicate detection (fast and effective)
            return str(imagehash.average_hash(image, hash_size=16))
        except Exception as e:
            logger.error(f"Error computing hash for {image_path}: {e}")
            return None
    
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

            # Request more results to account for duplicates we'll filter
            search_k = min(top_k * 5, len(self.metadata))
            distances, indices = self.index.search(query_embedding, search_k)

            # Compute query image hash for deduplication
            query_hash = self.compute_image_hash(query_image_path)

            results = []
            seen_hashes = set()

            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if len(results) >= top_k:
                    break

                if idx < len(self.metadata):
                    metadata = self.metadata[idx]
                    # Use relative_path from metadata (already calculated during embedding generation)
                    relative_path = metadata.get("relative_path", metadata.get("image_path", ""))

                    # Compute image hash for deduplication
                    image_path = metadata.get("image_path", "")
                    img_hash = self.compute_image_hash(image_path)

                    # Skip if we've already seen this image (duplicate)
                    if img_hash and img_hash in seen_hashes:
                        continue

                    if img_hash:
                        seen_hashes.add(img_hash)

                    # Normalize to forward slashes
                    relative_path = relative_path.replace('\\', '/')
                    path_obj = Path(relative_path)
                    parent_path = str(path_obj.parent)
                    # Ensure location uses forward slashes
                    location = parent_path.replace('\\', '/') if parent_path != '.' else ""

                    # Calculate similarity score (0-100 scale)
                    # IndexFlatIP returns inner product (cosine similarity for normalized vectors)
                    # Range is [-1, 1], convert to [0, 100]
                    similarity_score = float((distance + 1) * 50)  # Convert [-1,1] to [0,100]
                    similarity_score = min(100.0, max(0.0, similarity_score))  # Clamp to [0,100]

                    result = {
                        "fullPath": relative_path,
                        "id": str(uuid.uuid4()),
                        "location": location,
                        "name": path_obj.stem,
                        "type": "file",
                        "similarity": round(similarity_score, 2),
                        "distance": float(distance)  # Raw distance for debugging
                    }

                    results.append(result)
            
            return {
                "success": True,
                "results": results,
                "total_results": len(results),
                "query_processed": True,
                "query_hash": query_hash,  # For debugging
                "request_id": str(uuid.uuid4()),  # Unique request ID
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

class CompanyManager:
    """Manages multiple company ImageSearchAPI instances with lazy loading."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.company_apis = {}
            self.api_lock = threading.Lock()
            self.embeddings_base = Path(EMBEDDINGS_BASE_DIR)

            # Initialize shared model (used by all companies)
            logger.info("Initializing shared MobileNetV3 model...")
            self.shared_model = models.mobilenet_v3_large(pretrained=True)
            self.shared_model = torch.nn.Sequential(*list(self.shared_model.children())[:-1])
            self.shared_model.eval()

            self.shared_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            logger.info("CompanyManager initialized successfully")
            self.initialized = True

    def _validate_company_embeddings(self, embeddings_path):
        """Check if company embeddings directory has all required files."""
        required_files = ['faiss_index.bin', 'metadata.json']
        for file in required_files:
            if not (embeddings_path / file).exists():
                return False, f"Missing required file: {file}"
        return True, None

    def list_available_companies(self):
        """Scan embeddings directory and return list of valid companies."""
        companies = []

        if not self.embeddings_base.exists():
            return companies

        for item in self.embeddings_base.iterdir():
            if item.is_dir():
                company_name = item.name
                is_valid, error = self._validate_company_embeddings(item)
                if is_valid:
                    companies.append({
                        'name': company_name,
                        'embeddings_path': str(item),
                        'loaded': company_name in self.company_apis
                    })
                else:
                    logger.warning(f"Invalid embeddings for {company_name}: {error}")

        return companies

    def get_company_api(self, company_name):
        """Get or create ImageSearchAPI instance for a company."""
        # Check if already loaded
        if company_name in self.company_apis:
            return self.company_apis[company_name], None

        # Thread-safe loading
        with self.api_lock:
            # Double-check after acquiring lock
            if company_name in self.company_apis:
                return self.company_apis[company_name], None

            # Validate company embeddings exist
            embeddings_path = self.embeddings_base / company_name

            if not embeddings_path.exists():
                return None, f"Company '{company_name}' not found"

            is_valid, error = self._validate_company_embeddings(embeddings_path)
            if not is_valid:
                return None, f"Invalid embeddings for '{company_name}': {error}"

            # For now, we don't use separate image directories - handled by frontend
            # But we could add a config file per company to specify this
            images_base_dir = embeddings_path  # Placeholder

            try:
                logger.info(f"Loading embeddings for company: {company_name}")
                api = ImageSearchAPI(
                    embeddings_path,
                    images_base_dir,
                    self.shared_model,
                    self.shared_transform
                )
                self.company_apis[company_name] = api
                logger.info(f"Successfully loaded {company_name} with {len(api.metadata)} images")
                return api, None
            except Exception as e:
                logger.error(f"Failed to load company {company_name}: {e}")
                return None, f"Failed to load company embeddings: {str(e)}"

    def get_stats(self):
        """Get statistics about loaded companies."""
        return {
            'total_companies_available': len(self.list_available_companies()),
            'loaded_companies': list(self.company_apis.keys()),
            'loaded_count': len(self.company_apis)
        }


# Initialize CompanyManager
try:
    company_manager = CompanyManager()
    logger.info("CompanyManager initialized successfully")
except Exception as e:
    logger.error(f"Error initializing CompanyManager: {e}")
    company_manager = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    manager_stats = company_manager.get_stats() if company_manager else {}
    return jsonify({
        "status": "healthy",
        "api_ready": company_manager is not None,
        "timestamp": datetime.now().isoformat(),
        **manager_stats
    })

@app.route('/api/companies', methods=['GET'])
def list_companies():
    """List all available companies."""
    if company_manager is None:
        return jsonify({
            "success": False,
            "error": "CompanyManager not initialized"
        }), 500

    companies = company_manager.list_available_companies()
    return jsonify({
        "success": True,
        "companies": companies,
        "total": len(companies)
    })

@app.route('/api/stats/<company_name>', methods=['GET'])
def get_company_stats(company_name):
    """Get stats for a specific company."""
    if company_manager is None:
        return jsonify({
            "success": False,
            "error": "CompanyManager not initialized"
        }), 500

    api, error = company_manager.get_company_api(company_name)
    if error:
        return jsonify({
            "success": False,
            "error": error
        }), 400

    return jsonify({
        "success": True,
        "company": company_name,
        "stats": {
            "total_images": len(api.metadata),
            "index_info": api.index_info
        }
    })

@app.route('/api/search/<company_name>', methods=['POST'])
def search_similar_images(company_name):
    """Search similar images for a specific company."""
    if company_manager is None:
        return jsonify({
            "success": False,
            "error": "CompanyManager not initialized"
        }), 500

    logger.info(f"=== NEW SEARCH REQUEST for {company_name} ===")

    # Get company API instance
    search_api, error = company_manager.get_company_api(company_name)
    if error:
        return jsonify({
            "success": False,
            "error": error
        }), 400

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
    if request.content_length and request.content_length > MAX_FILE_SIZE:
        return jsonify({
            "success": False,
            "error": f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        }), 400

    filepath = None
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

        # Add company field to response
        if results.get('success'):
            results['company'] = company_name

        # Normalize paths for Windows compatibility
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

        logger.info(f"Returning {len(results.get('results', []))} results with timestamp {results.get('timestamp')}")

        # Create response with no-cache headers
        response = jsonify(results)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response

    except Exception as e:
        # Clean up file if error occurs
        if filepath:
            try:
                os.remove(filepath)
            except:
                pass

        logger.error(f"Search failed for {company_name}: {e}")
        return jsonify({
            "success": False,
            "error": f"Search failed: {str(e)}"
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "success": False,
        "error": f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
    }), 413

if __name__ == '__main__':
    app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
    app.run(debug=True, host='0.0.0.0', port=5000)