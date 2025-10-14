import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import faiss
import json
import pickle
from pathlib import Path
import argparse
from tqdm import tqdm
import time
import hashlib

class MobileNetV3Embedder:
    """Extract embeddings using MobileNetV3."""
    
    def __init__(self):
        print("Loading MobileNetV3 model...")
        # Load pretrained MobileNetV3
        self.model = models.mobilenet_v3_large(pretrained=True)
        
        # Remove the final classification layer to get features
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        print("MobileNetV3 model loaded successfully!")
        print(f"Feature dimension: 960")
    
    def extract_embedding(self, image_path):
        """Extract embedding from a single image."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze()
            
            # Convert to numpy and normalize
            embedding = features.cpu().numpy()
            
            # L2 normalization for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

def find_all_images(root_dir, extensions=None):
    """Find all image files in directory and subdirectories."""
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    
    root_path = Path(root_dir)
    image_files = []
    
    print(f"Scanning directory: {root_dir}")
    for ext in extensions:
        image_files.extend(root_path.rglob(f"*{ext}"))
        image_files.extend(root_path.rglob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    print(f"Found {len(image_files)} image files")
    return image_files

def generate_embeddings(input_dir, output_dir, batch_size=32):
    """Generate embeddings for all images and build FAISS index."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = find_all_images(input_dir)
    
    if len(image_files) == 0:
        print("No images found!")
        return
    
    # Initialize embedder
    embedder = MobileNetV3Embedder()
    
    # Storage for embeddings and metadata
    embeddings_list = []
    metadata_list = []
    failed_files = []
    
    print(f"\nProcessing {len(image_files)} images...")
    
    # Process images
    start_time = time.time()
    for idx, image_path in enumerate(tqdm(image_files, desc="Extracting embeddings")):
        # Extract embedding
        embedding = embedder.extract_embedding(image_path)
        
        if embedding is not None:
            embeddings_list.append(embedding)
            
            # Store metadata
            relative_path = image_path.relative_to(input_dir)
            metadata = {
                'index': len(embeddings_list) - 1,
                'image_path': str(image_path),
                'relative_path': str(relative_path),
                'filename': image_path.name,
                'file_size': image_path.stat().st_size,
                'image_id': hashlib.sha256(str(image_path).encode()).hexdigest()[:16]
            }
            metadata_list.append(metadata)
        else:
            failed_files.append(str(image_path))
        
        # Progress update every 100 images
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (len(image_files) - idx - 1) / rate
            print(f"  Processed {idx + 1}/{len(image_files)} images "
                  f"({rate:.1f} img/s, ETA: {remaining/60:.1f} min)")
    
    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.2f} seconds ({len(embeddings_list)/total_time:.2f} img/s)")
    print(f"Successfully processed: {len(embeddings_list)} images")
    print(f"Failed: {len(failed_files)} images")
    
    if len(embeddings_list) == 0:
        print("No embeddings generated!")
        return
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings_list, dtype='float32')
    print(f"\nEmbeddings shape: {embeddings_array.shape}")
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    dimension = embeddings_array.shape[1]
    
    # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings_array)
    
    print(f"FAISS index built with {faiss_index.ntotal} vectors")
    
    # Save FAISS index
    index_path = output_path / "faiss_index.bin"
    faiss.write_index(faiss_index, str(index_path))
    print(f"FAISS index saved to: {index_path}")
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    # Save embeddings as numpy array (optional, for backup)
    embeddings_path = output_path / "embeddings.npy"
    np.save(embeddings_path, embeddings_array)
    print(f"Embeddings saved to: {embeddings_path}")
    
    # Save failed files list
    if failed_files:
        failed_path = output_path / "failed_files.txt"
        with open(failed_path, 'w') as f:
            f.write('\n'.join(failed_files))
        print(f"Failed files list saved to: {failed_path}")
    
    # Save index info
    info = {
        'total_images': len(embeddings_list),
        'failed_images': len(failed_files),
        'embedding_dimension': dimension,
        'model': 'MobileNetV3-Large',
        'index_type': 'FAISS IndexFlatIP',
        'input_directory': str(input_dir),
        'generation_time': total_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    info_path = output_path / "index_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Index info saved to: {info_path}")
    
    print("\n" + "="*60)
    print("EMBEDDING GENERATION COMPLETE!")
    print("="*60)
    print(f"Total images processed: {len(embeddings_list)}")
    print(f"Failed images: {len(failed_files)}")
    print(f"Embedding dimension: {dimension}")
    print(f"Output directory: {output_path}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Generate image embeddings using MobileNetV3 and build FAISS index'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing images (will search recursively)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./embeddings',
        help='Output directory for embeddings and FAISS index (default: ./embeddings)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("IMAGE EMBEDDING GENERATION")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: MobileNetV3-Large")
    print("="*60)
    
    # Check if input directory exists
    if not Path(args.input_dir).exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    # Generate embeddings
    generate_embeddings(args.input_dir, args.output_dir, args.batch_size)

if __name__ == "__main__":
    main()