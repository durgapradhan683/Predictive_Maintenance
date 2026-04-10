# Use a minimal base image with Python 3.9 installed
#FROM python:3.9

# Set the working directory inside the container to /app
#WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory
#COPY . .

# Install Python dependencies listed in requirements.txt
#RUN pip3 install -r requirements.txt

#RUN useradd -m -u 1000 user
#USER user
#	PATH=/home/user/.local/bin:$PATH

#WORKDIR $HOME/app

#COPY --chown=user . $HOME/app

# Define the command to run the Streamlit app on port "8501" and make it accessible externally
#CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]

FROM python:3.9-slim

# Create non-root user
RUN useradd -m -u 1000 user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# Copy application files
COPY --chown=user . .

# Switch to non-root user
USER user

# Install dependencies locally for the user
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
