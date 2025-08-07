FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Upgrade pip and install pip-tools first
RUN pip install --upgrade pip setuptools wheel
RUN pip install pip-tools

COPY pyproject.toml ./

# Compile requirements.txt from pyproject.toml using pip-tools
RUN pip-compile --output-file=requirements.txt pyproject.toml

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8050

CMD ["gunicorn", "--bind", "0.0.0.0:8050", "vertex.descriptive_dashboard:server"]
