"""
A real network-backed tool (wttr.in), demonstrating how to wrap an
external, unreliable dependency safely: short timeout, narrow except,
and a ToolError that the caller can see and reason about instead of
the whole agent run crashing.
"""

from __future__ import annotations

from typing import ClassVar

import httpx
from pydantic import BaseModel, Field

from agent_toolkit.tools.base import BaseTool, ToolError


class WeatherArgs(BaseModel):
    location: str = Field(..., description="City name, e.g. 'Kochi' or 'Berlin'")


class WeatherTool(BaseTool):
    name: ClassVar[str] = "get_weather"
    description: ClassVar[str] = "Get current weather conditions for a named city."
    args_model: ClassVar[type[BaseModel]] = WeatherArgs

    def run(self, location: str) -> dict:
        url = f"https://wttr.in/{location}"
        try:
            response = httpx.get(url, params={"format": "j1"}, timeout=8.0)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ToolError(f"Weather lookup for '{location}' timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise ToolError(
                f"Weather service returned {exc.response.status_code} for '{location}'"
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"Weather lookup failed: {exc}") from exc

        try:
            data = response.json()
            current = data["current_condition"][0]
        except (ValueError, KeyError, IndexError) as exc:
            raise ToolError("Weather service returned an unexpected format") from exc

        return {
            "location": location,
            "temp_c": current.get("temp_C"),
            "feels_like_c": current.get("FeelsLikeC"),
            "description": current.get("weatherDesc", [{}])[0].get("value"),
            "humidity_pct": current.get("humidity"),
        }
