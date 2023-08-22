FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the contents of the current directory into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Set the OPENAI_API_KEY environment variable
ENV NAME=OPENAI_API_KEY

# Define the command to run your Streamlit app
CMD ["streamlit", "run", "clinical_trial_app.py", "--server.port", "8561"]

