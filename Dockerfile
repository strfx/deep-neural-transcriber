# Dockerfile to run the Deep Neural Transcriber MVP.
FROM python:3.8

# Location of the Flask's main app
ENV FLASK_APP=src/dnt/ui/app.py

# Expose Flask service over port 8080
EXPOSE 8080

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg sox

# Install Deep Neural Transcriber
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python setup.py install

# Ensure tflite-version deepspeech is installed
RUN pip install deepspeech-tflite

# Run Flask app
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8080"]
