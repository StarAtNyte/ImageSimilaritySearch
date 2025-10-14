# Image Search API

Flask API for similar image search using pre-computed embeddings and FAISS index.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have generated embeddings:
```bash
python generate_embeddings.py --input_dir AOF --output_dir my_embeddings
```

3. For searching similar images using script:
```bash
python search_similar_images.py --query_image image.jpg --embeddings_dir my_embeddings
```

4. Run the API:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### POST /api/search
Upload an image and find similar images.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body:
  - `image`: Image file (PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP)
  - `top_k`: Number of results to return (1-50, default: 10)

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "rank": 1,
      "filename": "Adelaide.png",
      "relative_path": "Abstract/Adelaide.png",
      "image_url": "/images/Abstract/Adelaide.png",
      "similarity_score": 0.95,
      "similarity_percentage": 95.0,
      "file_size": 1234567
    }
  ],
  "total_results": 10,
  "query_processed": true,
  "timestamp": "2025-10-12T10:30:45.123456"
}
```

### GET /api/health
Check API status.

**Response:**
```json
{
  "status": "healthy",
  "api_ready": true,
  "timestamp": "2025-10-12T10:30:45.123456"
}
```

### GET /api/stats
Get index statistics.

**Response:**
```json
{
  "success": true,
  "stats": {
    "total_images": 1500,
    "index_info": {
      "total_images": 1500,
      "embedding_dimension": 960,
      "model": "MobileNetV3-Large"
    }
  }
}
```

### GET /images/<path>
Serve images from the dataset.

**Example:** `http://localhost:5000/images/Abstract/Adelaide.png`

## Usage Examples

### JavaScript/Frontend
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('top_k', 5);

fetch('http://localhost:5000/api/search', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  if (data.success) {
    data.results.forEach(result => {
      console.log(`${result.rank}. ${result.filename} (${result.similarity_percentage}%)`);
      // Display image: data.image_url
    });
  }
});
```

### cURL
```bash
curl -X POST http://localhost:5000/api/search \
  -F "image=@query_image.jpg" \
  -F "top_k=5"
```

### Python
```python
import requests

with open('query_image.jpg', 'rb') as f:
    files = {'image': f}
    data = {'top_k': 5}
    response = requests.post('http://localhost:5000/api/search', 
                           files=files, data=data)
    result = response.json()
    
if result['success']:
    for item in result['results']:
        print(f"{item['rank']}. {item['filename']} ({item['similarity_percentage']}%)")
```

## Configuration

- **MAX_FILE_SIZE**: 16MB upload limit
- **ALLOWED_EXTENSIONS**: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP
- **Host**: 0.0.0.0 (accessible from network)
- **Port**: 5000

## File Structure
```
ImageSearch/
├── app.py                 # Flask API
├── generate_embeddings.py # Generate embeddings script
├── search_similar_images.py # CLI search script
├── requirements.txt       # Dependencies
├── AOF/                  # Image dataset
├── my_embeddings/        # FAISS index and metadata
└── uploads/             # Temporary upload folder (auto-created)
```

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Success
- 400: Bad request (invalid file, missing parameters)
- 413: File too large
- 500: Server error

All error responses include a JSON object with `success: false` and an `error` message.