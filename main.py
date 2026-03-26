# Updated CORS Configuration

# CORS origins
CORS_ORIGINS = [
    "https://glistening-cannoli-6bfc3b.netlify.app",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    os.environ.get("FRONTEND_URL", ""),
]