import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import faiss
import json
import sys
from pathlib import Path
import argparse
from datetime import datetime
import uuid
import warnings

warnings.filterwarnings('ignore')

class ImageSearch:
    """A class to perform image similarity search."""
    
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
        
        # Initialize model for embedding extraction
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
        """Extracts a L2-normalized embedding from an image file."""
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
            print(f"Error extracting embedding: {e}", file=sys.stderr)
            return None
            
    def get_relative_path(self, absolute_path):
        """Converts an absolute file path to a relative path based on the image base directory."""
        abs_path = Path(absolute_path)
        try:
            relative = str(abs_path.relative_to(self.images_base_dir))
            return relative.replace('\\', '/')
        except ValueError:
            return abs_path.name

    def search(self, query_image_path, top_k=10):
        """
        Performs similarity search and returns results in a JSON format
        compatible with the Flask API response.
        """
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
            search_k = min(top_k * 3, len(self.metadata))
            distances, indices = self.index.search(query_embedding, search_k)

            results = []
            seen_filenames = set()  # Track seen filenames for deduplication

            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if len(results) >= top_k:
                    break

                if idx < len(self.metadata):
                    metadata = self.metadata[idx]

                    # Get relative path for response
                    relative_path = metadata.get("relative_path", metadata.get("image_path", ""))
                    # Normalize Windows backslashes to forward slashes for cross-platform compatibility
                    relative_path_normalized = relative_path.replace('\\', '/')
                    path_obj = Path(relative_path_normalized)

                    # Extract filename (without extension) for deduplication
                    filename_no_ext = path_obj.stem.lower()  # Case-insensitive comparison

                    # Skip if we've already seen this filename
                    if filename_no_ext in seen_filenames:
                        continue

                    # Track this filename
                    seen_filenames.add(filename_no_ext)

                    parent_path = str(path_obj.parent)
                    # Ensure location uses forward slashes
                    location = parent_path if parent_path != '.' else ""

                    # Calculate similarity score (0-100 scale)
                    # IndexFlatIP returns inner product (cosine similarity for normalized vectors)
                    # Range is [-1, 1], convert to [0, 100]
                    similarity_score = float((distance + 1) * 50)  # Convert [-1,1] to [0,100]
                    similarity_score = min(100.0, max(0.0, similarity_score))  # Clamp to [0,100]

                    result = {
                        "fullPath": relative_path_normalized,
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

def main():
    parser = argparse.ArgumentParser(
        description='Perform image similarity search and output results as JSON.'
    )
    parser.add_argument(
        '--query_image',
        type=str,
        required=True,
        help='Path to the query image.'
    )
    parser.add_argument(
        '--embeddings_dir',
        type=str,
        default=None,
        help='Directory containing the FAISS index and metadata. If not specified, will use embeddings/{company}'
    )
    parser.add_argument(
        '--company',
        type=str,
        default=None,
        help='Company name (used to auto-locate embeddings directory as embeddings/{company})'
    )
    parser.add_argument(
        '--images_base_dir',
        type=str,
        default=None,
        help='Base directory of the image collection. Defaults to embeddings directory.'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of similar images to find.'
    )

    args = parser.parse_args()

    # Determine embeddings directory
    if args.embeddings_dir:
        embeddings_dir = args.embeddings_dir
    elif args.company:
        embeddings_dir = f'./embeddings/{args.company}'
    else:
        print("Error: Either --embeddings_dir or --company must be specified", file=sys.stderr)
        print("Examples:", file=sys.stderr)
        print("  python search_similar_images.py --query_image test.jpg --company AOF", file=sys.stderr)
        print("  python search_similar_images.py --query_image test.jpg --embeddings_dir ./my_embeddings", file=sys.stderr)
        sys.exit(1)

    # Default images_base_dir to embeddings_dir if not specified
    images_base_dir = args.images_base_dir if args.images_base_dir else embeddings_dir

    try:
        searcher = ImageSearch(embeddings_dir, images_base_dir)
        results = searcher.search(args.query_image, top_k=args.top_k)

        # Add company field if company was specified
        if args.company and results.get('success'):
            results['company'] = args.company

        print(json.dumps(results, indent=2))

    except FileNotFoundError as e:
        error_response = {
            "success": False,
            "error": str(e),
        }
        print(json.dumps(error_response, indent=2), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_response = {
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}",
        }
        print(json.dumps(error_response, indent=2), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
