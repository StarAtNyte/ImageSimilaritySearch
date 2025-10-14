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
import warnings
import shutil
warnings.filterwarnings('ignore')

class ImageSearcherAPI:
    """API-friendly image searcher that outputs JSON and saves results."""
    
    def __init__(self, embeddings_dir, base_images_dir=None, results_dir="./results"):
        self.embeddings_dir = Path(embeddings_dir)
        self.base_images_dir = Path(base_images_dir) if base_images_dir else None
        self.results_dir = Path(results_dir)
        
        # Load FAISS index
        index_path = self.embeddings_dir / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = self.embeddings_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load index info
        info_path = self.embeddings_dir / "index_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.index_info = json.load(f)
        
        # Initialize embedder
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
        """Extract embedding from query image."""
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
            return None
    
    def get_relative_path(self, absolute_path):
        """Convert absolute path to relative path for API."""
        abs_path = Path(absolute_path)
        
        if self.base_images_dir:
            try:
                return str(abs_path.relative_to(self.base_images_dir))
            except ValueError:
                pass
        
        # Fallback: return filename only
        return abs_path.name
    
    def create_results_directory(self, query_image_path):
        """Create a timestamped results directory for this search."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_name = Path(query_image_path).stem
        session_dir = self.results_dir / f"{query_name}_{timestamp}"
        
        # Create directory structure
        session_dir.mkdir(parents=True, exist_ok=True)
        images_dir = session_dir / "similar_images"
        images_dir.mkdir(exist_ok=True)
        
        return session_dir, images_dir
    
    def copy_query_image(self, query_image_path, session_dir):
        """Copy the query image to results directory."""
        try:
            query_path = Path(query_image_path)
            if query_path.exists():
                dest_path = session_dir / f"query_{query_path.name}"
                shutil.copy2(query_path, dest_path)
                return str(dest_path.relative_to(self.results_dir))
        except Exception as e:
            print(f"Warning: Could not copy query image: {e}", file=sys.stderr)
        return None
    
    def copy_similar_images(self, results, images_dir):
        """Copy found similar images to results directory."""
        copied_images = []
        
        for result in results:
            try:
                # Get the full path from metadata
                image_path = result.get("full_path")
                if not image_path or not Path(image_path).exists():
                    continue
                
                # Create numbered filename
                rank = result["rank"]
                original_name = Path(image_path).name
                ext = Path(image_path).suffix
                new_name = f"{rank:02d}_{original_name}"
                dest_path = images_dir / new_name
                
                # Copy the image
                shutil.copy2(image_path, dest_path)
                copied_images.append({
                    "rank": rank,
                    "original_path": image_path,
                    "saved_as": new_name
                })
                
            except Exception as e:
                print(f"Warning: Could not copy image at rank {result['rank']}: {e}", file=sys.stderr)
                continue
        
        return copied_images
    
    def search(self, query_image_path, top_k=10, save_results=True):
        """Search for similar images and return JSON response."""
        try:
            # Extract query embedding
            query_embedding = self.extract_embedding(query_image_path)
            if query_embedding is None:
                return {
                    "success": False,
                    "error": "Failed to extract embedding from query image",
                    "results": [],
                    "total_results": 0,
                    "query_processed": False
                }
            
            # Search
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
                if idx < len(self.metadata):
                    metadata = self.metadata[idx]
                    result = {
                        "rank": i + 1,
                        "filename": metadata.get("filename", "unknown"),
                        "relative_path": self.get_relative_path(metadata.get("image_path", "")),
                        "full_path": metadata.get("image_path", ""),
                        "similarity_score": float(distance),
                        "similarity_percentage": float(distance * 100),
                        "file_size": metadata.get("file_size", 0)
                    }
                    results.append(result)
            
            response = {
                "success": True,
                "query_image": str(query_image_path),
                "results": results,
                "total_results": len(results),
                "query_processed": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save results if requested
            if save_results:
                session_dir, images_dir = self.create_results_directory(query_image_path)
                
                # Copy query image
                query_saved = self.copy_query_image(query_image_path, session_dir)
                if query_saved:
                    response["query_image_saved"] = query_saved
                
                # Copy similar images
                copied_images = self.copy_similar_images(results, images_dir)
                response["images_copied"] = len(copied_images)
                response["copied_images_details"] = copied_images
                
                # Save JSON results
                json_path = session_dir / "results.json"
                with open(json_path, 'w') as f:
                    json.dump(response, f, indent=2)
                
                response["results_saved_to"] = str(session_dir.relative_to(Path.cwd()))
                response["json_saved_to"] = str(json_path.relative_to(Path.cwd()))
            
            return response
            
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
        description='API-friendly image similarity search with result saving'
    )
    parser.add_argument(
        '--query_image',
        type=str,
        required=True,
        help='Path to query image'
    )
    parser.add_argument(
        '--embeddings_dir',
        type=str,
        default="./my_embeddings",
        help='Directory containing FAISS index and metadata (default: ./my_embeddings)'
    )
    parser.add_argument(
        '--base_images_dir',
        type=str,
        help='Base directory for converting absolute paths to relative paths'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default="./results",
        help='Directory to save search results (default: ./results)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of similar images to return (default: 10)'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do not save results to disk (JSON output only)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize searcher
        searcher = ImageSearcherAPI(
            args.embeddings_dir, 
            args.base_images_dir,
            args.results_dir
        )
        
        # Perform search
        results = searcher.search(
            args.query_image, 
            args.top_k,
            save_results=not args.no_save
        )
        
        # Output JSON
        print(json.dumps(results, indent=2))
        
        # Print summary to stderr so it doesn't interfere with JSON output
        if results.get("success") and not args.no_save:
            print(f"\n✓ Results saved to: {results.get('results_saved_to')}", file=sys.stderr)
            print(f"✓ Found {results.get('total_results')} similar images", file=sys.stderr)
            print(f"✓ Copied {results.get('images_copied', 0)} images", file=sys.stderr)
        
    except Exception as e:
        # Error response in JSON format
        error_response = {
            "success": False,
            "error": str(e),
            "results": [],
            "total_results": 0,
            "query_processed": False
        }
        print(json.dumps(error_response, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()