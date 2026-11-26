# # Use the official Python image based on Alpine Linux 3.12
# FROM python:3.11-alpine

# # Install system dependencies for OpenCV wheels (check OpenCV documentation for exact versions)
# # RUN apk add --no-cache py3-numpy py3-matplotlib

# # Copy . /app
# COPY . /app

# # Set the working directory in the container
# WORKDIR /app

# # Install the dependencies specified in requirements.txt (excluding OpenCV)
# # Assuming you have other dependencies besides OpenCV
# RUN pip install -r requirements.txt

# # RUN pip install opencv-python-headless
# # Install OpenCV using pre-built wheels

# # Run the Streamlit app
# CMD streamlit run stramlit_interface.py

# Use the official Python image based on Alpine Linux 3.12
FROM python:3.12

# Install system dependencies for OpenCV wheels (check OpenCV documentation for exact versions)
# RUN apk add --no-cache py3-numpy py3-matplotlib

# Copy . /app
COPY . /app

# Set the working directory in the container
WORKDIR /app

# Install the dependencies specified in requirements.txt (excluding OpenCV)
# Assuming you have other dependencies besides OpenCV
RUN pip install -r requirements.txt

# Install OpenCV using pre-built wheels
RUN pip install opencv-python

# Run the Streamlit app
CMD streamlit run stramlit_interface.py
