"""Manifest client for ColliderML datasets.

Provides fetching, validation, and selection utilities over the portal manifest.
Schema (minimal initial version):

{
  "campaigns": {
    "taster": { "default": true, "datasets": { ... } },
    "full_pileup_pilot": { "default": false, "datasets": { ... } }
  }
}

Within a campaign, per-dataset versions:
{
  "datasets": {
    "ttbar": {
      "default_version": "v1",
      "versions": {
        "v1": {
          "objects": {
            "hits": [ {"path": "...", "start_event": 0, "end_event": 999}, ... ]
          }
        }
      }
    }
  }
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import requests
from .config import MANIFEST_URL


@dataclass
class ManifestFile:
    path: str
    start_event: int
    end_event: int


class ManifestClient:
    def __init__(self, url: str = MANIFEST_URL, request_timeout_seconds: int = 10):
        self._url = url
        self._timeout = request_timeout_seconds
        self._cached: Optional[Dict] = None

    def refresh(self, force: bool = False) -> Dict:
        if self._cached is not None and not force:
            return self._cached
        resp = requests.get(self._url, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        self._validate(data)
        self._cached = data
        return data

    def _validate(self, data: Dict) -> None:
        if not isinstance(data, dict) or "campaigns" not in data:
            raise ValueError("Invalid manifest: missing campaigns")
        if not isinstance(data["campaigns"], dict) or not data["campaigns"]:
            raise ValueError("Invalid manifest: empty campaigns")

    def get_default_campaign_name(self) -> str:
        data = self.refresh()
        campaigns = data["campaigns"]
        for name, info in campaigns.items():
            if isinstance(info, dict) and info.get("default"):
                return name
        # Fallback to first campaign in dict order
        return next(iter(campaigns.keys()))

    def select_files(
        self,
        campaign: Optional[str],
        datasets: Optional[List[str]],
        objects: Optional[List[str]],
        max_events: Optional[int],
        version: Optional[str] = None,
    ) -> List[ManifestFile]:
        data = self.refresh()
        campaigns = data["campaigns"]
        campaign_name = campaign or self.get_default_campaign_name()
        if campaign_name not in campaigns:
            raise ValueError(f"Unknown campaign: {campaign_name}")
        camp = campaigns[campaign_name]
        ds_map = camp.get("datasets", {})

        selected: List[ManifestFile] = []

        ds_iter = datasets or list(ds_map.keys())
        for ds in ds_iter:
            if ds not in ds_map:
                # Skip silently to keep lightweight behavior; could warn later
                continue
            # Version selection per dataset
            ds_info = ds_map[ds]
            if not isinstance(ds_info, dict):
                continue
            if "versions" in ds_info:
                ver_map = ds_info.get("versions", {})
                default_ver = ds_info.get("default_version")
                chosen_version = version or default_ver
                if not chosen_version or chosen_version not in ver_map:
                    # Fallback: pick first available version
                    if ver_map:
                        chosen_version = next(iter(ver_map.keys()))
                    else:
                        continue
                ver_info = ver_map[chosen_version] or {}
                obj_map = ver_info.get("objects", {})
            else:
                # Backward compatibility: allow objects at dataset level
                obj_map = ds_info.get("objects", {})
            obj_iter = objects or list(obj_map.keys())
            for obj in obj_iter:
                files = obj_map.get(obj, [])
                # Apply max_events per object type so that selecting ALL objects
                # includes each object independently up to the requested events
                remaining_for_object = max_events if max_events is not None else None
                for fi in files:
                    mf = ManifestFile(
                        path=fi["path"],
                        start_event=int(fi.get("start_event", 0)),
                        end_event=int(fi.get("end_event", 0)),
                    )
                    selected.append(mf)
                    if remaining_for_object is not None:
                        events_in_file = mf.end_event - mf.start_event + 1
                        remaining_for_object -= events_in_file
                        if remaining_for_object <= 0:
                            break
        return selected


__all__ = ["ManifestClient", "ManifestFile"]


