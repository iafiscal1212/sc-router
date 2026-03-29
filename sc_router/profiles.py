"""Model quality profiles based on LLM EKG health monitoring.

Maintains per-(model, sc_level) quality profiles so the router can
recommend the best model for each complexity level.

Requires: pip install sc-router[ekg]  (llm-ekg >= 1.1.0)
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

_DEFAULT_STORAGE_DIR = os.path.join(
    os.path.expanduser("~"), ".sc_router", "profiles"
)

# Minimum responses before a profile is considered reliable
_MIN_RESPONSES = 5


def _require_ekg():
    """Import and return LLMAnalyzer, raising clear error if missing."""
    try:
        from llm_ekg import LLMAnalyzer
        return LLMAnalyzer
    except ImportError:
        raise ImportError(
            "llm-ekg is required for model profiles. "
            "Install with: pip install sc-router[ekg]"
        )


class ProfileManager:
    """Manages LLM quality profiles per (model, sc_level) pair.

    Each profile wraps an LLMAnalyzer that tracks response quality
    over time. The manager can recommend the best model for a given
    SC level based on accumulated EKG scores.

    Usage:
        pm = ProfileManager()
        pm.record("gpt-4", sc_level=2, response="...", response_time_s=1.2)
        pm.record("claude-3", sc_level=2, response="...", response_time_s=0.8)
        best = pm.best_model(sc_level=2)  # returns model with highest EKG score
    """

    def __init__(self, storage_dir: str = _DEFAULT_STORAGE_DIR):
        self._storage_dir = storage_dir
        self._profiles: Dict[str, Dict[str, Any]] = {}
        # key = "{model}__sc{level}", value = {"analyzer": LLMAnalyzer, "model": str, "sc_level": int}
        self.load()

    @staticmethod
    def _key(model: str, sc_level: int) -> str:
        """Build profile key from model name and SC level."""
        safe_model = model.replace("/", "_").replace("\\", "_")
        return f"{safe_model}__sc{sc_level}"

    def _get_or_create(self, model: str, sc_level: int):
        """Get existing profile or create a new one."""
        LLMAnalyzer = _require_ekg()
        key = self._key(model, sc_level)
        if key not in self._profiles:
            self._profiles[key] = {
                "analyzer": LLMAnalyzer(),
                "model": model,
                "sc_level": sc_level,
            }
        return self._profiles[key]

    def record(
        self,
        model: str,
        sc_level: int,
        response: str,
        response_time_s: float = 0.0,
    ) -> Dict:
        """Record a model response and update its quality profile.

        Args:
            model: Model identifier (e.g. "gpt-4", "claude-3-opus").
            sc_level: SC complexity level (0-3) of the query.
            response: The model's response text.
            response_time_s: Response time in seconds.

        Returns:
            Dict with anomaly_score, global_score, n_responses.
        """
        profile = self._get_or_create(model, sc_level)
        analyzer = profile["analyzer"]

        result = analyzer.ingest(
            response=response,
            timestamp=time.time(),
            response_time_s=response_time_s,
        )

        summary = analyzer.get_summary()
        return {
            "anomaly_score": result["state"]["anomaly_score"],
            "global_score": summary["global_score_100"],
            "n_responses": summary["n_responses"],
        }

    def best_model(
        self,
        sc_level: int,
        candidates: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Return the model with the best EKG score for a given SC level.

        Args:
            sc_level: SC complexity level to query.
            candidates: Optional list of model names to consider.
                If None, considers all models with profiles at this level.

        Returns:
            Model name with highest global_score_100, or None if no
            profile has >= _MIN_RESPONSES responses.
        """
        _require_ekg()
        best_score = -1
        best_model_name = None

        for key, profile in self._profiles.items():
            if profile["sc_level"] != sc_level:
                continue
            if candidates and profile["model"] not in candidates:
                continue

            summary = profile["analyzer"].get_summary()
            if summary["n_responses"] < _MIN_RESPONSES:
                continue

            score = summary["global_score_100"]
            if score > best_score:
                best_score = score
                best_model_name = profile["model"]

        return best_model_name

    def get_profile(self, model: str, sc_level: int) -> Optional[Dict]:
        """Get summary for a specific (model, sc_level) profile.

        Returns:
            Dict with model, sc_level, global_score_100, verdict,
            hallucination_risk, n_responses — or None if no profile exists.
        """
        key = self._key(model, sc_level)
        if key not in self._profiles:
            return None

        profile = self._profiles[key]
        summary = profile["analyzer"].get_summary()
        return {
            "model": profile["model"],
            "sc_level": profile["sc_level"],
            "global_score_100": summary["global_score_100"],
            "verdict": summary["verdict"],
            "hallucination_risk": summary["hallucination_risk"],
            "n_responses": summary["n_responses"],
            "trend": summary.get("trend", "stable"),
        }

    def all_profiles(self) -> List[Dict]:
        """Return summaries for all profiles."""
        result = []
        for key in sorted(self._profiles.keys()):
            profile = self._profiles[key]
            summary = profile["analyzer"].get_summary()
            result.append({
                "model": profile["model"],
                "sc_level": profile["sc_level"],
                "global_score_100": summary["global_score_100"],
                "verdict": summary["verdict"],
                "hallucination_risk": summary["hallucination_risk"],
                "n_responses": summary["n_responses"],
                "trend": summary.get("trend", "stable"),
            })
        return result

    def save(self):
        """Persist all profiles to disk as JSON files."""
        os.makedirs(self._storage_dir, exist_ok=True)

        for key, profile in self._profiles.items():
            analyzer = profile["analyzer"]
            # Convert numpy arrays to lists for JSON serialization
            fh = [
                f.tolist() if hasattr(f, 'tolist') else list(f)
                for f in analyzer.feature_history
            ]
            data = {
                "model": profile["model"],
                "sc_level": profile["sc_level"],
                "feature_history": fh,
                "n_responses": len(fh),
            }
            filepath = os.path.join(self._storage_dir, f"{key}.json")
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

    def load(self):
        """Load profiles from disk, reconstructing LLMAnalyzers.

        Replays saved feature vectors through the engine to reconstruct
        the full internal state (Hebbian weights, momentum, historials).
        """
        if not os.path.isdir(self._storage_dir):
            return

        try:
            LLMAnalyzer = _require_ekg()
        except ImportError:
            return

        for filename in os.listdir(self._storage_dir):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(self._storage_dir, filename)
            try:
                with open(filepath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            model = data.get("model", "")
            sc_level = data.get("sc_level", 0)
            feature_history = data.get("feature_history", [])

            if not model or not feature_history:
                continue

            # Reconstruct analyzer by replaying feature vectors directly
            # through the internal engine (bypasses text extraction).
            import numpy as np
            analyzer = LLMAnalyzer()
            for i, features in enumerate(feature_history):
                fv = np.asarray(features, dtype=np.float64)
                # Feed features directly to engine
                analyzer.feature_history.append(fv)
                analyzer.timestamps.append(float(i))
                analyzer.metadata.append({})
                ir = analyzer._engine.step(fv)
                analyzer.state_history.append(ir)
                # Multiscale analysis (same logic as ingest)
                fr = None
                if len(analyzer.feature_history) >= analyzer._freq.fft_window:
                    fr = analyzer._freq.analyze_all(
                        np.array(analyzer.feature_history)
                    )
                analyzer.scale_history.append(fr)

            key = self._key(model, sc_level)
            self._profiles[key] = {
                "analyzer": analyzer,
                "model": model,
                "sc_level": sc_level,
            }

    def reset(
        self,
        model: Optional[str] = None,
        sc_level: Optional[int] = None,
    ):
        """Reset profile(s).

        - reset() — reset all
        - reset(model="gpt-4") — reset all levels for gpt-4
        - reset(sc_level=2) — reset all models for SC(2)
        - reset(model="gpt-4", sc_level=2) — reset specific profile
        """
        keys_to_remove = []
        for key, profile in self._profiles.items():
            if model is not None and profile["model"] != model:
                continue
            if sc_level is not None and profile["sc_level"] != sc_level:
                continue
            keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._profiles[key]
            filepath = os.path.join(self._storage_dir, f"{key}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
