FROM python:3.11-slim

# Install system dependencies for audio, visuals, and building extensions
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    portaudio19-dev \
    libasound2-dev \
    libsdl2-mixer-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy and install your Python libraries
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt \
    && pip install --no-cache-dir voila ipywidgets

# Copy all your chatbot code files into the container
COPY . .

# Expose the mandatory port for Hugging Face Spaces
EXPOSE 7860

# Serve your notebook directly through Voilà in a dark theme
CMD ["voila", "--host=0.0.0.0", "--port=7860", "--no-browser", "--theme=dark", "app.ipynb"]
