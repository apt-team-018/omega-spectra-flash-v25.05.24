
# Use the official Python image as a parent image
FROM python:3.10.14

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the app directory is in the PYTHONPATH
ENV PYTHONPATH=/app

# Define environment variables (required)
ENV MODEL_PATH=${MODEL_PATH}
ENV IMAGE_UPLOAD_ENDPOINT=${IMAGE_UPLOAD_ENDPOINT}
ENV IMAGE_UPLOAD_APIKEY=${IMAGE_UPLOAD_APIKEY}

# Define optional environment variables
ENV API_KEYS=${API_KEYS:-""}
ENV TOKEN=${TOKEN:-""}

# Validate and set QUEUE_BATCH_SIZE as an integer
ARG DEFAULT_QUEUE_BATCH_SIZE=10
ENV QUEUE_BATCH_SIZE=${QUEUE_BATCH_SIZE}

# Expose the port that the app runs on
EXPOSE 8080

# Define the command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080",  "--workers", "12"]