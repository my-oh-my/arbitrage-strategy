# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables to prevent Python from writing .pyc files and to run in unbuffered mode
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies if needed (e.g., for certain Python packages that need compilation)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Copy the requirements file and install dependencies first
# This leverages Docker's layer caching to speed up builds if requirements haven't changed.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application's source code from the 'src' directory into the container
COPY ./src .

# Command to run the application when the container starts
CMD ["python", "main.py"]
