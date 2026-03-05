FROM python:3.11-slim

WORKDIR /app

# Needed to pip install from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends git \
  && rm -rf /var/lib/apt/lists/*

# If you have requirements.txt, keep this.
# If you don't, you can remove these two lines.
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install fin-kit from GitHub
RUN pip install --no-cache-dir git+https://github.com/lakshya-aga/fin-kit.git

# Copy your app
COPY . .

# If your MCP server entrypoint is different, change this.
CMD ["python", "mcp_server.py"]
