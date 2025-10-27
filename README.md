# Image Similarity Search

This project provides a set of tools to perform image similarity searches with **multi-tenant support**. It uses a pre-trained MobileNetV3 model to generate embeddings for images and FAISS for efficient similarity searching. The project includes scripts for generating embeddings, a command-line interface (CLI) for searching, and a Flask-based web API to expose the search functionality.

## Features

- **Multi-Tenant Architecture**: Support multiple companies/clients with isolated embeddings
- **Embedding Generation**: Traverses a directory of images and generates vector embeddings
- **Similarity Search**: Finds the most similar images to a given query image
- **Lazy Loading**: Company embeddings load on-demand for efficient memory usage
- **Shared Model**: Single MobileNetV3 model shared across all companies (memory efficient)
- **Command-Line Interface**: CLI tools for both embedding generation and search
- **REST API**: Flask server with multi-tenant endpoints
- **Thread-Safe**: Concurrent request handling with thread-safe company loading

## Project Structure

```
ImageSearch/
│
├── .gitignore
├── app.py                   # Flask web application with multi-tenant API
├── generate_embeddings.py   # Script to generate image embeddings per company
├── search_similar_images.py # CLI tool for performing similarity searches
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── embeddings/              # Base directory for all company embeddings
│   ├── AOF/                 # Company 1 embeddings
│   │   ├── faiss_index.bin
│   │   ├── metadata.json
│   │   ├── index_info.json
│   │   └── embeddings.npy
│   │
│   ├── CompanyB/            # Company 2 embeddings
│   │   ├── faiss_index.bin
│   │   ├── metadata.json
│   │   ├── index_info.json
│   │   └── embeddings.npy
│   │
│   └── ...                  # Additional companies
│
└── uploads/                 # Temporary storage for images uploaded to the API
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ImageSearch
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

There are three main parts to this project: generating embeddings per company, running the API server, and using the CLI for searches.

### 1. Generate Embeddings for a Company

Before you can search for similar images, you need to generate embeddings for each company's image collection.

**Recommended approach (using `--company` flag):**

```bash
python generate_embeddings.py --input_dir ./path/to/AOF/images --company AOF
python generate_embeddings.py --input_dir ./path/to/CompanyB/images --company CompanyB
```

This will automatically create embeddings in:
- `embeddings/AOF/`
- `embeddings/CompanyB/`

**Alternative approach (custom output directory):**

```bash
python generate_embeddings.py --input_dir ./images --output_dir ./custom/path
```

**Parameters:**
- `--input_dir`: Path to the directory containing images (will search recursively)
- `--company`: Company name (auto-generates output to `embeddings/{company}/`)
- `--output_dir`: Custom output directory (overrides `--company`)
- `--batch_size`: Batch size for processing (default: 32)

**Note:** Either `--company` or `--output_dir` must be specified.

### 2. Run the Web API

To start the Flask server with multi-tenant support:

```bash
python app.py
```

The server will start on `http://0.0.0.0:5000` by default.

**How it works:**
- The server initializes with a shared MobileNetV3 model
- Company embeddings are loaded **on-demand** when first requested
- Once loaded, company data stays cached in memory for fast subsequent searches
- Thread-safe for concurrent requests across multiple companies

### 3. Use the CLI for Search

You can perform searches directly from your terminal using the `search_similar_images.py` script.

**Recommended approach (using `--company` flag):**

```bash
python search_similar_images.py --query_image ./test.jpg --company AOF --top_k 10
python search_similar_images.py --query_image ./test.jpg --company CompanyB --top_k 5
```

**Alternative approach (custom embeddings directory):**

```bash
python search_similar_images.py --query_image ./test.jpg --embeddings_dir ./custom/path --top_k 10
```

**Parameters:**
- `--query_image`: Path to the query image (required)
- `--company`: Company name (auto-locates `embeddings/{company}/`)
- `--embeddings_dir`: Custom embeddings directory (overrides `--company`)
- `--images_base_dir`: Base directory of image collection (optional)
- `--top_k`: Number of similar images to return (default: 10)

The script outputs a JSON object containing the search results.

## API Endpoints

The Flask application (`app.py`) provides the following multi-tenant endpoints:

### Health Check

- **URL**: `/api/health`
- **Method**: `GET`
- **Description**: Checks the health of the API and returns loaded company statistics.
- **Success Response (200)**:
  ```json
  {
    "status": "healthy",
    "api_ready": true,
    "timestamp": "2025-10-27T12:00:00.000Z",
    "total_companies_available": 3,
    "loaded_companies": ["AOF", "CompanyB"],
    "loaded_count": 2
  }
  ```

### List Available Companies

- **URL**: `/api/companies`
- **Method**: `GET`
- **Description**: Returns a list of all available companies with valid embeddings.
- **Success Response (200)**:
  ```json
  {
    "success": true,
    "companies": [
      {
        "name": "AOF",
        "embeddings_path": "embeddings/AOF",
        "loaded": true
      },
      {
        "name": "CompanyB",
        "embeddings_path": "embeddings/CompanyB",
        "loaded": false
      }
    ],
    "total": 2
  }
  ```

### Get Company Statistics

- **URL**: `/api/stats/<company_name>`
- **Method**: `GET`
- **Description**: Returns statistics for a specific company's image index.
- **URL Parameters**:
  - `company_name`: The name of the company (e.g., `AOF`, `CompanyB`)
- **Example**: `/api/stats/AOF`
- **Success Response (200)**:
  ```json
  {
    "success": true,
    "company": "AOF",
    "stats": {
      "total_images": 12345,
      "index_info": {
        "total_images": 12345,
        "embedding_dimension": 960,
        "model": "MobileNetV3-Large",
        "timestamp": "2025-10-27T10:00:00"
      }
    }
  }
  ```
- **Error Response (400)**:
  ```json
  {
    "success": false,
    "error": "Company 'InvalidCompany' not found"
  }
  ```

### Search for Similar Images

- **URL**: `/api/search/<company_name>`
- **Method**: `POST`
- **Description**: Upload an image to find the most similar images from a specific company's indexed collection.
- **URL Parameters**:
  - `company_name`: The name of the company (e.g., `AOF`, `CompanyB`)
- **Example**: `/api/search/AOF`
- **Form Data**:
  - `image`: The image file to query (required)
  - `top_k` (optional): The number of results to return (default: 10, max: 50)
- **Success Response (200)**:
  ```json
  {
    "success": true,
    "company": "AOF",
    "results": [
      {
        "fullPath": "Abstract/Adelaide.png",
        "id": "a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
        "location": "Abstract",
        "name": "Adelaide",
        "type": "file"
      }
    ],
    "total_results": 1,
    "query_processed": true,
    "timestamp": "2025-10-27T12:05:00.000Z"
  }
  ```
- **Error Response (400)**:
  ```json
  {
    "success": false,
    "error": "Company 'InvalidCompany' not found"
  }
  ```
- **Error Response (400)** - Invalid file:
  ```json
  {
    "success": false,
    "error": "File type not allowed. Allowed types: png, jpg, jpeg, gif, bmp, tiff, tif, webp"
  }
  ```

## Multi-Tenant Architecture

### How It Works

The application uses a **CompanyManager** singleton class to manage multiple company instances efficiently:

1. **Shared Model**: A single MobileNetV3 model is loaded once and shared across all companies, significantly reducing memory usage.

2. **Lazy Loading**: Company-specific FAISS indexes and metadata are loaded only when first requested, not at startup.

3. **Memory-Efficient Caching**: Once a company is loaded, its data stays in memory for fast subsequent searches.

4. **Thread-Safe**: The CompanyManager uses locks to ensure thread-safe loading and access in concurrent environments.

### Directory Naming Convention

Each company must have embeddings stored in the following format:
```
embeddings/{company_name}/
```

For example:
- `embeddings/AOF/`
- `embeddings/Acme/`
- `embeddings/TechCorp/`

### Required Files per Company

Each company embeddings directory must contain:
- `faiss_index.bin` - FAISS index file
- `metadata.json` - Image metadata
- `index_info.json` - Index information (optional but recommended)
- `embeddings.npy` - Raw embeddings (optional, for backup)

## Example Workflows

### Adding a New Company

1. **Generate embeddings for the new company:**
   ```bash
   python generate_embeddings.py --input_dir ./images/NewCompany --company NewCompany
   ```

2. **Verify the company is available:**
   ```bash
   curl http://localhost:5000/api/companies
   ```

3. **Test search for the new company:**
   ```bash
   curl -X POST -F "image=@test.jpg" http://localhost:5000/api/search/NewCompany
   ```

### Searching Across Multiple Companies

```bash
# Search in Company A
curl -X POST -F "image=@query.jpg" -F "top_k=5" \
  http://localhost:5000/api/search/AOF

# Search in Company B
curl -X POST -F "image=@query.jpg" -F "top_k=5" \
  http://localhost:5000/api/search/CompanyB
```

### Monitoring Loaded Companies

```bash
# Check which companies are currently loaded in memory
curl http://localhost:5000/api/health
```

## Migration from Single-Tenant Setup

If you have an existing single-tenant setup with embeddings in a directory like `my_embeddings/`, you can migrate to the multi-tenant structure:

1. **Create the embeddings directory:**
   ```bash
   mkdir -p embeddings
   ```

2. **Move existing embeddings to company-specific folder:**
   ```bash
   mv my_embeddings embeddings/AOF
   ```

3. **Update configuration in app.py (if needed):**
   The new code automatically uses `embeddings/` as the base directory.

4. **Test the migration:**
   ```bash
   python app.py
   curl http://localhost:5000/api/companies
   curl http://localhost:5000/api/stats/AOF
   ```

## Technical Details

### Model Information
- **Model**: MobileNetV3-Large (pretrained on ImageNet)
- **Embedding Dimension**: 960
- **Normalization**: L2 normalization for cosine similarity

### FAISS Index
- **Index Type**: IndexFlatIP (Inner Product)
- **Distance Metric**: Cosine similarity (with normalized vectors)

### Performance Considerations
- **Memory Usage**: Each company's FAISS index uses approximately `num_images * 960 * 4 bytes`
- **Loading Time**: Initial load of a company takes 1-5 seconds depending on index size
- **Search Speed**: Typically <100ms for indexes up to 100K images

### Supported Image Formats
- PNG, JPG, JPEG, GIF, BMP, TIFF, TIF, WEBP

### File Size Limits
- Maximum upload size: 16 MB per image

## Development Notes

### Production Deployment Recommendations

⚠️ **Important**: The current setup uses Flask's development server. For production deployment:

1. **Use a production WSGI server** (Gunicorn, uWSGI)
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Set appropriate CORS policies** - Currently allows all origins

3. **Add authentication/authorization** - No auth is currently implemented

4. **Configure proper logging** - Set up CloudWatch or similar for AWS

5. **Add rate limiting** - Prevent abuse

6. **Use environment variables** for configuration

7. **Set up health checks** for load balancers

8. **Consider using GPU** for faster embedding extraction (if available)

## Troubleshooting

### Company not found error
- Verify the company embeddings directory exists: `embeddings/{company}_embeddings/`
- Check that `faiss_index.bin` and `metadata.json` exist in the directory
- Ensure directory naming follows the `{name}_embeddings` convention

### Out of memory errors
- Reduce the number of loaded companies by restarting the server
- Consider implementing LRU cache eviction for companies
- Use a machine with more RAM or optimize FAISS index

### Slow first search
- This is expected - the company's embeddings are being loaded
- Subsequent searches will be much faster (cached)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
