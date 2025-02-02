# ---------- Stage 1: Builder ----------
FROM python:3.9-slim as builder

# Set the working directory in the builder container.
WORKDIR /app

# Install build dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file and install dependencies.
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---------- Stage 2: Production ----------
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from the builder stage.
COPY --from=builder /install /usr/local

# Copy all files from the build context into /app.
COPY /lib /app/lib
COPY /services /app/services

# (Optional) Verify that main.py exists.
RUN ls -la /app

# Expose the port your app uses.
EXPOSE 8080

# Declare a volume mount for persistent storage.
VOLUME /price

# Run the application.
CMD ["python", "services/price_service/main.py"]