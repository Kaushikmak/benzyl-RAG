#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   RAG Web Interface Startup Script${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if required files exist
if [ ! -f "api.py" ] || [ ! -f "streamlit_app.py" ]; then
    echo -e "${RED}Error: api.py or streamlit_app.py not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python found${NC}"
echo -e "${GREEN}✓ Required files found${NC}\n"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${BLUE}Shutting down services...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo -e "${BLUE}Starting FastAPI backend...${NC}"
python3 api.py > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
echo -e "${BLUE}Waiting for backend to initialize...${NC}"
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Backend failed to start. Check backend.log${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Backend running (PID: $BACKEND_PID)${NC}"
echo -e "${GREEN}✓ API available at http://localhost:8000${NC}\n"

# Start frontend
echo -e "${BLUE}Starting Streamlit frontend...${NC}"
streamlit run streamlit_app.py > frontend.log 2>&1 &
FRONTEND_PID=$!

sleep 3

# Check if frontend is running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Frontend failed to start. Check frontend.log${NC}"
    kill $BACKEND_PID
    exit 1
fi

echo -e "${GREEN}✓ Frontend running (PID: $FRONTEND_PID)${NC}"
echo -e "${GREEN}✓ UI available at http://localhost:8501${NC}\n"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Both services are running!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "Access the web interface at: ${GREEN}http://localhost:8501${NC}"
echo -e "API documentation at: ${GREEN}http://localhost:8000/docs${NC}\n"

echo -e "Logs:"
echo -e "  Backend: ${BLUE}backend.log${NC}"
echo -e "  Frontend: ${BLUE}frontend.log${NC}\n"

echo -e "${RED}Press Ctrl+C to stop both services${NC}\n"

# Wait for processes
wait