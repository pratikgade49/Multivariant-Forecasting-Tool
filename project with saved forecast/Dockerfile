FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Expose port
EXPOSE 5173

# Start development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# Stage 2: Python Backend with Frontend Static Files
FROM python:3.12-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to backend
WORKDIR /app/backend

# Copy backend requirements and install dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./

# Create directory for static files
RUN mkdir -p /app/static

# Copy frontend build from frontend-build stage
COPY --from=frontend-build /app/dist /app/static

# Create a simple script to serve static files
RUN echo 'from fastapi.staticfiles import StaticFiles\n\
# Add this to your FastAPI app in main.py\n\
app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")' > serve_static.py

# Make the start script executable
COPY backend/start.sh ./
RUN chmod +x start.sh

# Expose port
EXPOSE 8000

# Command to run the application with proper setup sequence
CMD ["./start.sh"]
