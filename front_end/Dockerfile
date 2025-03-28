# Use Node.js as base image
FROM node:20-slim

# Install pnpm
RUN npm install -g pnpm

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml ./

# Install dependencies
RUN pnpm install

# Copy the rest of the application
COPY . .

# Build the application
RUN pnpm build

# Expose the port the app runs on
EXPOSE 5173

# Set host environment variable for proper binding
ENV HOST=0.0.0.0
ENV PORT=5173

# Command to run the application
CMD ["node", "build/index.js"]