# Step 1: Use a base image
FROM python:3.10-slim

# Step 2: Set working directory in the container
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Step 3: Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy the model and app to the container
COPY src/artifacts/model.joblib .
COPY src/app.py .

# Step 5: Expose the port that Flask will run on
EXPOSE 5050

# Step 6: Run the Flask app when the container starts
CMD ["python", "app.py"]
