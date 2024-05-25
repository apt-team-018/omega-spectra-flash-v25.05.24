
## Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install git and system dependencies
RUN apt-get update && apt-get install -y git

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install the apt-diffusers package from a specific git branch
RUN pip install git+https://github.com/all-secure-src/apt-diffusers-v190524.git

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variables (required)
ENV MODEL_PATH=""
ENV IMAGE_UPLOAD_ENDPOINT=""
ENV IMAGE_UPLOAD_APIKEY=""
ENV UPLOAD_IMAGE_STATIC_PATH=""

# Define optional environment variables
ENV API_KEYS=""
ENV TOKEN=""

# Validate and set QUEUE_BATCH_SIZE as an integer
ENV QUEUE_BATCH_SIZE=10

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]