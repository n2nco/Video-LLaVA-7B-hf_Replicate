FROM python:3.10-slim

WORKDIR /src

RUN apt-get update && apt-get install -y ffmpeg

# We don't need to install Python or copy requirements here
# Cog will handle Python package installation

# We'll let Cog handle the Python package installation
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# We don't need to copy the code here, Cog will do that

# Remove the CMD instruction, Cog will handle running the predictor
# CMD ["python", "predict.py"]