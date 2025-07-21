# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size and is a best practice
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code (main.py, graph.py)
COPY . .

# Expose the port the app runs on. 
# Cloud Run provides a PORT environment variable, which uvicorn will use.
# 8080 is a common default if PORT isn't set.
EXPOSE 8080

# Command to run the application using uvicorn web server
# It will listen on all available network interfaces (0.0.0.0).
CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8080"]
