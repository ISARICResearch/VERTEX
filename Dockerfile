FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .
RUN pip install --upgrade pip
# Autograd-gamma does not have prebuild wheels for some reason so we need wheel
RUN pip install --upgrade wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

EXPOSE 8050

# Command to run the app using Gunicorn
# reload for dev work, mount the application code to the container

CMD ["gunicorn", "--reload", "--bind", "0.0.0.0:8050", "descriptive_dashboard:server"]