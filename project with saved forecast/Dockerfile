# Stage 1: Build Frontend
FROM node:20-alpine AS frontend-build

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code and build frontend
COPY . .
RUN npm run build

# Stage 2: Python Backend with Frontend Static Files
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app/backend

# Copy backend requirements and install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./

# Copy frontend build output to backend static directory
COPY --from=frontend-build /app/dist /app/backend/static

# Make the start script executable
COPY backend/start.sh ./
RUN chmod +x start.sh

# Expose backend port
EXPOSE 8000

# Start backend application
CMD ["./start.sh"]
