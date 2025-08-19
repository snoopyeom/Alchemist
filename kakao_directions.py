import os
import json
import time
import math
import hashlib
from typing import Tuple
from pathlib import Path

import requests


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


class KakaoDirectionsProvider:
    """도로 거리/시간 제공자.

    Kakao Mobility Directions API를 호출해 출발지-도착지 간
    도로 거리(km)와 시간(min)을 반환한다.
    내부적으로 디스크 캐시 및 재시도/백오프를 내장한다.
    실패 시 하버사인 거리 기반 값으로 폴백한다.
    """

    def __init__(self,
                 api_key: str | None = None,
                 priority: str | None = None,
                 summary: str | None = None,
                 cache_dir: str | None = None,
                 cache_ttl: float | None = None):
        self.api_key = api_key or os.getenv("KAKAO_API_KEY", "")
        self.priority = priority or os.getenv("KAKAO_PRIORITY", "TIME")
        self.summary = summary or os.getenv("KAKAO_SUMMARY_ONLY", "true")
        self.metric = os.getenv("ROAD_COST_METRIC", "time")
        self.cache_dir = Path(cache_dir or os.getenv("CACHE_DIR", ".kakao_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ttl = cache_ttl or os.getenv("CACHE_TTL")
        self.cache_ttl = float(ttl) if ttl is not None else None

    def _cache_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _load_cache(self, key: str) -> Tuple[float, float] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        if self.cache_ttl is not None:
            if time.time() - path.stat().st_mtime > self.cache_ttl:
                return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data["km"]), float(data["min"])
        except Exception:
            return None

    def _save_cache(self, key: str, km: float, minutes: float) -> None:
        path = self._cache_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"km": km, "min": minutes}, f)
        except Exception:
            pass

    def pair_cost(self, lat1: float, lon1: float, lat2: float, lon2: float,
                   max_retries: int = 3) -> Tuple[float, float]:
        """좌표 쌍을 받아 (거리_km, 시간_min)을 반환."""
        key = f"{lon1:.6f},{lat1:.6f}->{lon2:.6f},{lat2:.6f}"
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        if not self.api_key:
            km = _haversine(lat1, lon1, lat2, lon2)
            minutes = (km / 40.0) * 60.0
            return km, minutes
        url = "https://apis-navi.kakaomobility.com/v1/directions"
        headers = {
            "Authorization": f"KakaoAK {self.api_key}",
            "Content-Type": "application/json",
        }
        params = {
            "origin": f"{lon1},{lat1}",
            "destination": f"{lon2},{lat2}",
            "priority": self.priority,
            "summary": self.summary,
        }
        backoff = 0.5
        for attempt in range(max_retries + 1):
            try:
                r = requests.get(url, headers=headers, params=params, timeout=5)
                if r.status_code == 200:
                    routes = r.json().get("routes", [])
                    if routes:
                        summary = routes[0].get("summary", {})
                        dist = float(summary.get("distance", 0.0)) / 1000.0
                        dur = float(summary.get("duration", 0.0)) / 60.0
                        self._save_cache(key, dist, dur)
                        return dist, dur
                if r.status_code in (401, 403, 429, 400):
                    time.sleep(backoff)
                    backoff *= 2
                    continue
            except requests.exceptions.RequestException:
                time.sleep(backoff)
                backoff *= 2
                continue
        km = _haversine(lat1, lon1, lat2, lon2)
        minutes = (km / 40.0) * 60.0
        self._save_cache(key, km, minutes)
        return km, minutes
