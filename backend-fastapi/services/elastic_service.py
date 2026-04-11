"""
ARIHANT SOC - Elasticsearch Service
===================================
Handles logging and querying of security events
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import os

# Try importing Elasticsearch
try:
    from elasticsearch import Elasticsearch, AsyncElasticsearch
    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False
    print("[WARN] Elasticsearch library not available. Using in-memory logging.")


class ElasticService:
    """
    Elasticsearch integration for logging predictions and alerts
    
    Falls back to in-memory storage if Elasticsearch is not available
    """
    
    # Index names
    INDEX_NETWORK = "arihant-network-logs"
    INDEX_APPLICATION = "arihant-application-logs"
    INDEX_AUDIO = "arihant-audio-logs"
    INDEX_HUMAN = "arihant-human-logs"
    INDEX_ALERTS = "arihant-alerts"
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9200,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_ssl: bool = False
    ):
        self.host = host
        self.port = port
        self.es_client = None
        self._connected = False
        
        # In-memory fallback storage
        self._memory_store: Dict[str, List[Dict]] = {
            self.INDEX_NETWORK: [],
            self.INDEX_APPLICATION: [],
            self.INDEX_AUDIO: [],
            self.INDEX_HUMAN: [],
            self.INDEX_ALERTS: []
        }
        
        if ES_AVAILABLE:
            try:
                es_url = f"{'https' if use_ssl else 'http'}://{host}:{port}"
                
                if username and password:
                    self.es_client = Elasticsearch(
                        [es_url],
                        basic_auth=(username, password),
                        verify_certs=False
                    )
                else:
                    self.es_client = Elasticsearch([es_url])
                
                # Test connection
                if self.es_client.ping():
                    self._connected = True
                    print(f"[ES] Connected to Elasticsearch at {es_url}")
                    self._create_indices()
                else:
                    print(f"[ES] Failed to connect to Elasticsearch at {es_url}")
                    
            except Exception as e:
                print(f"[ES] Elasticsearch connection error: {e}")
                self._connected = False
    
    def _create_indices(self):
        """Create indices with mappings if they don't exist"""
        if not self._connected or not self.es_client:
            return
        
        # Common mapping for all indices
        base_mapping = {
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "prediction": {"type": "keyword"},
                    "confidence": {"type": "float"},
                    "severity": {"type": "keyword"},
                    "source_ip": {"type": "ip"},
                    "details": {"type": "object", "enabled": False}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }
        
        for index in self._memory_store.keys():
            try:
                if not self.es_client.indices.exists(index=index):
                    self.es_client.indices.create(index=index, body=base_mapping)
                    print(f"[ES] Created index: {index}")
            except Exception as e:
                print(f"[ES] Failed to create index {index}: {e}")
    
    async def log_prediction(
        self,
        index: str,
        input_data: Dict[str, Any],
        prediction: Dict[str, Any],
        source_ip: Optional[str] = None
    ) -> bool:
        """
        Log a prediction to Elasticsearch
        
        Args:
            index: Index name (use class constants)
            input_data: Input features/data
            prediction: Prediction result
            source_ip: Source IP if available
            
        Returns:
            Success status
        """
        document = {
            "timestamp": datetime.utcnow().isoformat(),
            "input": input_data,
            "prediction": prediction.get("attack_type") or prediction.get("prediction"),
            "confidence": prediction.get("confidence", 0),
            "severity": prediction.get("severity", "UNKNOWN"),
            "is_attack": prediction.get("is_attack") or prediction.get("is_fake") or prediction.get("phishing", False),
            "source_ip": source_ip,
            "processing_time_ms": prediction.get("processing_time_ms"),
            "details": prediction
        }
        
        if self._connected and self.es_client:
            try:
                self.es_client.index(index=index, document=document)
                return True
            except Exception as e:
                print(f"[ES] Failed to log to {index}: {e}")
        
        # Fallback to memory
        if index in self._memory_store:
            self._memory_store[index].append(document)
            # Keep only last 10000 entries per index
            if len(self._memory_store[index]) > 10000:
                self._memory_store[index] = self._memory_store[index][-10000:]
        
        return True
    
    async def log_alert(self, alert: Dict[str, Any]) -> bool:
        """Log an alert"""
        document = {
            "timestamp": datetime.utcnow().isoformat(),
            **alert
        }
        
        if self._connected and self.es_client:
            try:
                self.es_client.index(index=self.INDEX_ALERTS, document=document)
                return True
            except Exception as e:
                print(f"[ES] Failed to log alert: {e}")
        
        self._memory_store[self.INDEX_ALERTS].append(document)
        return True
    
    async def search_logs(
        self,
        index: str,
        query: Optional[Dict[str, Any]] = None,
        time_range: Optional[str] = "24h",
        size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search logs in Elasticsearch
        
        Args:
            index: Index to search
            query: Elasticsearch query (optional)
            time_range: Time range (e.g., "24h", "7d")
            size: Maximum results
            
        Returns:
            List of matching documents
        """
        # Parse time range
        now = datetime.utcnow()
        if time_range.endswith("h"):
            hours = int(time_range[:-1])
            start_time = now - timedelta(hours=hours)
        elif time_range.endswith("d"):
            days = int(time_range[:-1])
            start_time = now - timedelta(days=days)
        else:
            start_time = now - timedelta(hours=24)
        
        if self._connected and self.es_client:
            try:
                search_query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"range": {"timestamp": {"gte": start_time.isoformat()}}}
                            ]
                        }
                    },
                    "sort": [{"timestamp": {"order": "desc"}}],
                    "size": size
                }
                
                if query:
                    search_query["query"]["bool"]["must"].append(query)
                
                result = self.es_client.search(index=index, body=search_query)
                return [hit["_source"] for hit in result["hits"]["hits"]]
                
            except Exception as e:
                print(f"[ES] Search failed: {e}")
        
        # Fallback to memory search
        if index in self._memory_store:
            filtered = [
                doc for doc in self._memory_store[index]
                if datetime.fromisoformat(doc["timestamp"]) >= start_time
            ]
            return sorted(filtered, key=lambda x: x["timestamp"], reverse=True)[:size]
        
        return []
    
    async def get_stats(self, index: str, time_range: str = "24h") -> Dict[str, Any]:
        """Get statistics for an index"""
        logs = await self.search_logs(index, time_range=time_range, size=10000)
        
        if not logs:
            return {
                "total": 0,
                "attacks": 0,
                "by_severity": {},
                "by_type": {}
            }
        
        attacks = sum(1 for log in logs if log.get("is_attack", False))
        
        by_severity = {}
        by_type = {}
        
        for log in logs:
            sev = log.get("severity", "UNKNOWN")
            by_severity[sev] = by_severity.get(sev, 0) + 1
            
            pred = log.get("prediction", "Unknown")
            by_type[pred] = by_type.get(pred, 0) + 1
        
        return {
            "total": len(logs),
            "attacks": attacks,
            "by_severity": by_severity,
            "by_type": by_type
        }
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Elasticsearch"""
        return self._connected


# Global instance
elastic_service = ElasticService(
    host=os.getenv("ES_HOST", "localhost"),
    port=int(os.getenv("ES_PORT", "9200")),
    username=os.getenv("ES_USERNAME"),
    password=os.getenv("ES_PASSWORD")
)
