# Image Similarity Search

This project provides a set of tools to perform image similarity searches. It uses a pre-trained MobileNetV3 model to generate embeddings for images and Faiss for efficient similarity searching. The project includes scripts for generating embeddings, a command-line interface (CLI) for searching, and a Flask-based web API to expose the search functionality.

## Features

- **Embedding Generation**: Traverses a directory of images and generates vector embeddings.
- **Similarity Search**: Finds the most similar images to a given query image.
- **Command-Line Interface**: A CLI to perform searches directly from the terminal.
- **REST API**: A Flask server that exposes the search functionality through a simple API.
- **Configurable**: Easily configure image directories, embedding storage, and search parameters.

## Project Structure

```
ImageSearch/
│
├── .gitignore
├── app.py                   # Flask web application with API endpoints
├── generate_embeddings.py   # Script to generate image embeddings
├── search_similar_images.py # CLI tool for performing similarity searches
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
├── AOF/                     # Directory containing the image collection
│   └── ...
│
├── my_embeddings/           # Directory where embeddings and index are stored
│   ├── embeddings.npy
│   ├── faiss_index.bin
│   ├── index_info.json
│   └── metadata.json
│
├── results/                 # Default directory for saved search results (if modified to save)
│   └── ...
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

There are three main parts to this project: generating embeddings, running the API server, and using the CLI for searches.

### 1. Generate Embeddings

Before you can search for similar images, you need to generate embeddings for your image collection.

```bash
python generate_embeddings.py --images_dir AOF --embeddings_dir my_embeddings
```

- `--images_dir`: The path to the directory containing your images (e.g., `AOF`).
- `--embeddings_dir`: The path to the directory where the generated embeddings and index will be saved (e.g., `my_embeddings`).

This process will create the `my_embeddings` directory (if it doesn't exist) and save the `faiss_index.bin`, `metadata.json`, and other necessary files.

### 2. Run the Web API

To start the Flask server and use the REST API, run the following command:

```bash
python app.py
```

The server will start on `http://0.0.0.0:5000` by default.

### 3. Use the CLI for Search

You can also perform searches directly from your terminal using the `search_similar_images.py` script.

```bash
python search_similar_images.py --query_image /path/to/your/image.png --top_k 10
```

- `--query_image`: The path to the image you want to find similar images for.
- `--top_k`: The number of similar images to return.

The script will output a JSON object to the console containing the search results, in the same format as the `/api/search` endpoint.

## API Endpoints

The Flask application (`app.py`) provides the following endpoints:

### Health Check

- **URL**: `/api/health`
- **Method**: `GET`
- **Description**: Checks the health of the API and confirms if the search functionality is ready.
- **Success Response (200)**:
  ```json
  {
    "status": "healthy",
    "api_ready": true,
    "timestamp": "2025-10-14T12:00:00.000Z"
  }
  ```

### Get Index Statistics

- **URL**: `/api/stats`
- **Method**: `GET`
- **Description**: Returns statistics about the loaded image index.
- **Success Response (200)**:
  ```json
  {
    "success": true,
    "stats": {
      "total_images": 12345,
      "index_info": {
        "model_name": "mobilenet_v3_large",
        "embedding_dim": 960,
        "total_images": 12345,
        "timestamp": "2025-10-14T10:00:00.000Z"
      }
    }
  }
  ```

### Search for Similar Images

- **URL**: `/api/search`
- **Method**: `POST`
- **Description**: Upload an image to find the most similar images from the indexed collection.
- **Form Data**:
  - `image`: The image file to query.
  - `top_k` (optional): The number of results to return (default: 10, max: 50).
- **Success Response (200)**:
  ```json
  {
    "success": true,
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
    "timestamp": "2025-10-14T12:05:00.000Z"
  }
  ```
- **Error Response (400, 500)**:
  ```json
  {
    "success": false,
    "error": "Error message describing the issue."
  }
  ```
