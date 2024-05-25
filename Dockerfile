
# Use the official Python image as a parent image
FROM python:3.10.14

# Set the working directory
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the working directory
COPY app /app/app

# Expose the port that the app runs on
EXPOSE 8080

# Define environment variables (required)
ENV MODEL_PATH=${MODEL_PATH}
ENV IMAGE_UPLOAD_ENDPOINT=${IMAGE_UPLOAD_ENDPOINT}
ENV IMAGE_UPLOAD_APIKEY=${IMAGE_UPLOAD_APIKEY}

# Define optional environment variables
ENV API_KEYS=${API_KEYS:-""}
ENV TOKEN=${TOKEN:-""}

# Define the command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080",  "--workers", "20"]