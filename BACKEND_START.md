# How to Start the Backend

## Prerequisites

You need Python 3.11+ with pip installed on your system.

## Installation Steps

### 1. Navigate to Backend Directory
```bash
cd backend
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

Or if using pip3:
```bash
pip3 install -r requirements.txt
```

Or using python3 module:
```bash
python3 -m pip install -r requirements.txt
```

### 3. Verify Environment File
Make sure `backend/.env` exists with Supabase credentials (already created):
```bash
cat .env
```

You should see:
```
SUPABASE_URL=https://fgfpsqvzxsfgztejfzvj.supabase.co
SUPABASE_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
MODEL_VERSION=v1.0.0
```

## Starting the Backend

### Option 1: Using uvicorn directly
```bash
cd backend
uvicorn app.main:app --reload
```

### Option 2: With host and port specified
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 3: Without reload (production mode)
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Expected Output

When the backend starts successfully, you should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345] using StatReload
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## Verify Backend is Running

### Check Health Endpoint
```bash
curl http://localhost:8000/health
```

Should return:
```json
{"status":"healthy"}
```

### Check API Root
```bash
curl http://localhost:8000/
```

Should return:
```json
{
  "message": "GeoRAG API",
  "version": "1.0.0",
  "endpoints": {
    "hazards": "/api/v1/hazards",
    "active_learning": "/api/v1/active-learning",
    "missions": "/api/v1/missions",
    "models": "/api/v1/models"
  }
}
```

### View API Documentation
Open in browser:
```
http://localhost:8000/docs
```

This shows interactive Swagger UI with all endpoints.

## Test with Sample Data

### Get Hazard Areas
```bash
curl http://localhost:8000/api/v1/hazards
```

### Get Missions
```bash
curl http://localhost:8000/api/v1/missions
```

### Get Active Learning Stats
```bash
curl http://localhost:8000/api/v1/active-learning/statistics
```

## Common Issues

### Issue: "No module named 'fastapi'"
**Solution**: Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Issue: "No module named 'app'"
**Solution**: Make sure you're in the backend directory
```bash
cd backend
uvicorn app.main:app --reload
```

### Issue: Port 8000 already in use
**Solution**: Use a different port
```bash
uvicorn app.main:app --port 8001 --reload
```

Or kill the process using port 8000:
```bash
# Find process
lsof -ti:8000
# Kill it
kill -9 $(lsof -ti:8000)
```

### Issue: "Connection refused" when accessing API
**Solution**: Make sure backend is running
```bash
# Check if process is running
ps aux | grep uvicorn

# Check if port is listening
netstat -an | grep 8000
```

## Environment Variables

The backend uses these environment variables from `backend/.env`:

- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Anon/public key for client operations
- `SUPABASE_SERVICE_ROLE_KEY`: Service role key for admin operations
- `MODEL_VERSION`: Current model version (default: v1.0.0)

## Development Tips

### Enable Debug Logging
```bash
uvicorn app.main:app --reload --log-level debug
```

### Auto-reload on File Changes
The `--reload` flag automatically restarts the server when code changes.

### View Request Logs
All API requests are logged to console when running with uvicorn.

## Next Steps

Once the backend is running:

1. **Test API**: Visit `http://localhost:8000/docs` for interactive API documentation
2. **Start Frontend**: In another terminal, run `npm run dev` from project root
3. **Access Application**: Open `http://localhost:5173` in your browser
4. **View Data**: Navigate to Dashboard or Risk Map to see sample data

## Full Stack Running

You need two terminals:

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn app.main:app --reload
```

**Terminal 2 - Frontend:**
```bash
npm run dev
```

Then open `http://localhost:5173` in your browser.

The frontend will automatically connect to the backend API at `http://localhost:8000`.
