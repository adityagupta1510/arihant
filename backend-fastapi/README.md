# ARIHANT SOC - FastAPI Backend

AI-Driven Security Operations Center API with ML-powered threat detection.

## Features

- **Network Intrusion Detection** - LSTM-based network flow analysis
- **Application Layer Attack Detection** - SQL injection, XSS, command injection detection
- **Audio Deepfake Detection** - HAV-DF model for synthetic audio detection
- **Phishing Detection** - Email content analysis for social engineering
- **Real-time Alerts** - WebSocket-based alert broadcasting
- **Elasticsearch Integration** - Comprehensive logging and analytics

## Quick Start

### 1. Install Dependencies

```bash
cd backend-fastapi
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python run.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access the API

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/health
- **WebSocket**: ws://localhost:8000/ws/alerts

## API Endpoints

### Network Detection
- `POST /api/network/detect` - Analyze network flow
- `POST /api/network/detect/batch` - Batch analysis
- `GET /api/network/stats` - Detection statistics

### Application Detection
- `POST /api/application/detect` - Analyze request payload
- `POST /api/application/analyze-payload` - Quick pattern analysis
- `GET /api/application/attack-patterns` - List detectable patterns

### Audio Detection
- `POST /api/audio/detect` - Upload and analyze audio file
- `POST /api/audio/detect/features` - Analyze pre-extracted features
- `GET /api/audio/supported-formats` - List supported formats

### Human Threat Detection
- `POST /api/human/detect` - Analyze email for phishing
- `POST /api/human/analyze-quick` - Quick indicator scan
- `POST /api/human/analyze-sender` - Analyze sender address

### Alert Management
- `POST /api/alerts/create` - Create new alert
- `GET /api/alerts` - List alerts with filtering
- `GET /api/alerts/stats` - Alert statistics
- `PATCH /api/alerts/{id}` - Update alert
- `POST /api/alerts/{id}/resolve` - Resolve alert

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:
- `ELASTICSEARCH_URL` - Elasticsearch connection (optional)
- `MODELS_DIR` - Path to trained models
- `CORS_ORIGINS` - Allowed frontend origins

## Models

The backend expects trained models in the `../models/` directory:

- `nids_lstm_model.h5` - Network LSTM model
- `app_layer_lstm_model.h5` - Application LSTM model
- `best_model.pt` - Audio HAV-DF model
- `rf_model.pkl` - Random Forest fallback
- Various scalers and encoders (`.pkl` files)

## WebSocket Alerts

Connect to `ws://localhost:8000/ws/alerts` for real-time alerts:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/alerts');

ws.onmessage = (event) => {
  const alert = JSON.parse(event.data);
  console.log('Alert:', alert);
};
```

Alert message format:
```json
{
  "type": "alert",
  "data": {
    "id": "uuid",
    "attack_type": "DDoS",
    "severity": "CRITICAL",
    "confidence": 0.97,
    "source": "NETWORK",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Performance

- Target response time: <300ms
- Supports batch processing
- Async model inference
- Connection pooling for Elasticsearch

## License

Proprietary - ARIHANT SOC
