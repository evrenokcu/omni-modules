# ---------- Stage 1: Builder ----------
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies (e.g., gcc for compiling any packages if needed)
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install Python dependencies into a temporary location
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---------- Stage 2: Production ----------
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy the application source code
COPY main.py .

# Expose the default port (make sure it matches your application's default or desired runtime port)
EXPOSE 8000

# Run the application using the Python script so that it picks up the PORT env variable
CMD ["python", "main.py"]