
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

# Expose the port that the app runs on
EXPOSE 8080

# Define environment variables (required)
ENV MODEL_PATH=${MODEL_PATH}
ENV IMAGE_UPLOAD_ENDPOINT=${IMAGE_UPLOAD_ENDPOINT}
ENV IMAGE_UPLOAD_APIKEY=${IMAGE_UPLOAD_APIKEY}

# Define optional environment variables
ENV API_KEYS=${API_KEYS:-""}
ENV TOKEN=${TOKEN:-""}
ENV WORKERS=${WORKERS:-20}

# Validate and set QUEUE_BATCH_SIZE as an integer
ARG DEFAULT_QUEUE_BATCH_SIZE=4
ENV QUEUE_BATCH_SIZE=${QUEUE_BATCH_SIZE}

RUN if ! echo "$QUEUE_BATCH_SIZE" | grep -E '^[0-9]+$'; then \
        echo "Invalid QUEUE_BATCH_SIZE value, setting to default ${DEFAULT_QUEUE_BATCH_SIZE}"; \
        QUEUE_BATCH_SIZE=${DEFAULT_QUEUE_BATCH_SIZE}; \
    fi && \
    export QUEUE_BATCH_SIZE

# Define the command to run the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers ${WORKERS}"]