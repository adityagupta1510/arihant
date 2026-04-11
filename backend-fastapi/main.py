"""
ARIHANT SOC - FastAPI Backend
=============================
Production-ready AI-driven Security Operations Center API
Integrates ML models for Network, Application, Audio, and Human threat detection
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add parent directory to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from routes import network, application, audio, human, alerts
from services.model_loader import ModelLoader
from services.websocket_manager import ConnectionManager

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODELS_DIR = Path(__file__).parent.parent / "models"
API_VERSION = "1.0.0"
API_TITLE = "ARIHANT SOC API"
API_DESCRIPTION = """
## AI-Driven Security Operations Center

ARIHANT provides enterprise-grade threat detection across multiple layers:

- **Network Layer**: LSTM-based intrusion detection
- **Application Layer**: Attack pattern recognition  
- **Audio Layer**: Deepfake/spoof detection
- **Human Layer**: Phishing and social engineering detection

### Real-time Capabilities
- WebSocket alerts for instant threat notifications
- Elasticsearch integration for comprehensive logging
- Sub-300ms response times for all endpoints
"""

# ═══════════════════════════════════════════════════════════════════════════════
# LIFESPAN MANAGEMENT (Model Loading)
# ═══════════════════════════════════════════════════════════════════════════════

model_loader: Optional[ModelLoader] = None
ws_manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup at shutdown"""
    global model_loader
    
    print("\n" + "="*60)
    print("  ARIHANT SOC - Initializing AI Models")
    print("="*60)
    
    model_loader = ModelLoader(MODELS_DIR)
    await model_loader.load_all_models()
    
    # Store in app state for route access
    app.state.model_loader = model_loader
    app.state.ws_manager = ws_manager
    
    print("\n" + "="*60)
    print("  ✅ All models loaded successfully!")
    print("  🚀 ARIHANT SOC API is ready")
    print("="*60 + "\n")
    
    yield
    
    # Cleanup
    print("\n[INFO] Shutting down ARIHANT SOC...")
    if model_loader:
        model_loader.cleanup()

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*"  # Allow all in development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# INCLUDE ROUTERS
# ═══════════════════════════════════════════════════════════════════════════════

app.include_router(network.router, prefix="/api/network", tags=["Network Detection"])
app.include_router(application.router, prefix="/api/application", tags=["Application Detection"])
app.include_router(audio.router, prefix="/api/audio", tags=["Audio Detection"])
app.include_router(human.router, prefix="/api/human", tags=["Human Threat Detection"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alert Management"])

# ═══════════════════════════════════════════════════════════════════════════════
# ROOT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    """API Root - Health Check"""
    return {
        "service": "ARIHANT SOC API",
        "status": "ONLINE",
        "version": API_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "network": "/api/network/detect",
            "application": "/api/application/detect",
            "audio": "/api/audio/detect",
            "human": "/api/human/detect",
            "alerts": "/api/alerts",
            "websocket": "/ws/alerts"
        }
    }

@app.get("/api/health", tags=["Health"])
async def health_check():
    """Detailed Health Check"""
    models_status = {}
    if model_loader:
        models_status = model_loader.get_status()
    
    return {
        "status": "HEALTHY",
        "service": "ARIHANT SOC API",
        "version": API_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": None,  # Would need startup time tracking
        "models": models_status,
        "websocket_connections": ws_manager.active_connections_count
    }

@app.get("/api/stats", tags=["Statistics"])
async def get_stats():
    """Get API Statistics"""
    return {
        "total_predictions": alerts.alert_store.get_total_count(),
        "active_alerts": len(alerts.alert_store.alerts),
        "models_loaded": len(model_loader.loaded_models) if model_loader else 0,
        "websocket_clients": ws_manager.active_connections_count,
        "timestamp": datetime.utcnow().isoformat()
    }

# ═══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alerts
    
    Clients receive JSON messages:
    {
        "type": "alert",
        "data": {
            "id": "...",
            "attack_type": "DDoS",
            "severity": "CRITICAL",
            "confidence": 0.97,
            "source": "network",
            "timestamp": "..."
        }
    }
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            # Echo back for ping/pong
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     █████╗ ██████╗ ██╗██╗  ██╗ █████╗ ███╗   ██╗████████╗ ║
    ║    ██╔══██╗██╔══██╗██║██║  ██║██╔══██╗████╗  ██║╚══██╔══╝ ║
    ║    ███████║██████╔╝██║███████║███████║██╔██╗ ██║   ██║    ║
    ║    ██╔══██║██╔══██╗██║██╔══██║██╔══██║██║╚██╗██║   ██║    ║
    ║    ██║  ██║██║  ██║██║██║  ██║██║  ██║██║ ╚████║   ██║    ║
    ║    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ║
    ║                                                           ║
    ║           AI-Driven Security Operations Center            ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
