# This file contains configuration for PyActiveStorage.

# Force True for S3 exploratory tests
USE_S3 = True

# URL of Reductionist S3 Active Storage server.
S3_ACTIVE_STORAGE_URL = "http://localhost:8080"

# Path to a CA certificate for Reductionist server.
S3_ACTIVE_STORAGE_CACERT = None

# URL of S3 object store.
S3_URL = "http://localhost:9000"

# S3 access key / username.
S3_ACCESS_KEY = "minioadmin"

# S3 secret key / password.
S3_SECRET_KEY = "minioadmin"

# S3 bucket.
S3_BUCKET = "pyactivestorage"
