# Use Node.js 20.11.1 as specified in the documentation
FROM node:20.11.1-slim

# Install pnpm
RUN npm install -g pnpm

# Set working directory
WORKDIR /app/front_end

# Copy package files
COPY package.json pnpm-lock.yaml ./

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PORT=5173
ENV HOST=0.0.0.0
ENV NODE_ENV=production
ENV NODE_OPTIONS=--max-old-space-size=4096

# Build the application with environment variables
ARG VITE_BACKEND_URL
ENV VITE_BACKEND_URL=$VITE_BACKEND_URL

# Build with production optimizations
RUN pnpm build

# Install serve instead of http-server (more lightweight)
RUN npm install -g serve

# Create a script to handle graceful shutdown
RUN echo '#!/bin/sh\n\ntrap "kill -TERM \$PID" TERM INT\n\n# Start serve in the background\nserve -s dist -l 5173 &\nPID=\$!\n\n# Wait for the process\nwait \$PID' > /start.sh && chmod +x /start.sh

# Expose the port the app runs on
EXPOSE 5173

# Command to serve the built application with graceful shutdown
CMD ["/start.sh"]