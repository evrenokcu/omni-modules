# Stage 1: Build stage
FROM python:3.9-slim AS builder

# Create a working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies into a virtual environment in the /venv directory
RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.9-slim

# Create a working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /venv /venv

# Copy the application code
COPY app_async.py .

# Ensure the virtual environment's Python and pip are used
ENV PATH="/venv/bin:$PATH"

# Expose the port
EXPOSE 8080

# Set the entrypoint to run the Flask app
CMD ["python", "app_async.py"]