# Use the official slim Python image from the Docker Hub
FROM python:3.11.4-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install virtual environment
RUN pip install virtualenv

# Copy the requirements file into the container
COPY requirements.txt /app

# Create and activate a virtual environment, and install dependencies
RUN virtualenv venv && \
    . venv/bin/activate && \
    pip install -U pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 10234

# Command to run the application
CMD ["sh", "-c", ". venv/bin/activate && uvicorn main:app --host 0.0.0.0 --port 10234"]
