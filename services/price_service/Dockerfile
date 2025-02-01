# ---------- Stage 1: Builder ----------
FROM python:3.9-slim as builder

# Set the working directory
WORKDIR /app

# Install build dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file and install dependencies to a temporary location
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---------- Stage 2: Production ----------
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from the builder stage into the production image
COPY --from=builder /install /usr/local

# Copy the entire application code into the image
COPY . .

# Expose the default port (8000)
EXPOSE 8000

# Declare a volume mount at /price for persistent storage
VOLUME /price

# Run the application
CMD ["python", "main.py"]