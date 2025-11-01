# Use an official Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the src folder into the container
COPY src/ ./src/

# Command to run your Python script when the container starts
CMD ["python", "src/code/main.py"]
