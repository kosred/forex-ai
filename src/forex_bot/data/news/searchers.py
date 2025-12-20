import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

import requests

from ...domain.errors import APIBillingError

logger = logging.getLogger(__name__)


class PerplexitySearcher:
    def __init__(self, settings) -> None:
        self.available = False
        self.api_key = None
        self.endpoint = "https://api.perplexity.ai/chat/completions"
        key_env = getattr(settings.news, "perplexity_api_key_env", "PPLX_API_KEY")
        api_key = os.environ.get(key_env)
        if not api_key:
            try:
                secrets_path = getattr(settings, "secrets_file", "keys.txt")
                if secrets_path and os.path.exists(secrets_path):
                    with open(secrets_path, encoding="utf-8") as f:
                        for line in f:
                            if "=" in line:
                                k, v = line.split("=", 1)
                                if k.strip() == key_env:
                                    api_key = v.strip()
                                    break
            except Exception:
                api_key = None
        if not api_key:
            return
        self.api_key = api_key
        self.available = True
        self.model = getattr(settings.news, "perplexity_model", "sonar")

    def search(self, query: str, num_results: int = 10) -> list[dict[str, Any]]:
        if not self.available or not self.api_key:
            return []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a news search engine. "
                    "Return results as a valid JSON list of objects with keys: "
                    "'title', 'url', 'date' (ISO8601), 'snippet'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Find the latest {num_results} news headlines for: {query}. "
                    "Return ONLY the JSON list, no markdown, no preamble."
                ),
            },
        ]
        candidates = [self.model]
        fallback_models = [
            "sonar",
            "sonar-pro",
            "sonar-reasoning",
            "sonar-reasoning-pro",
            "sonar-deep-research",
        ]
        for m in fallback_models:
            if m not in candidates:
                candidates.append(m)

        for model_name in candidates:
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.1,  # Low temp for format adherence
            }
            try:
                resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)

                if resp.status_code in (401, 402, 403, 429):
                    raise APIBillingError(f"Perplexity API Critical Error: {resp.status_code} - {resp.text[:100]}")

                if resp.status_code == 400 and "Invalid model" in resp.text:
                    logger.info(f"Perplexity model '{model_name}' rejected; trying fallback...")
                    continue

                if resp.status_code != 200:
                    logger.info(f"Perplexity HTTP {resp.status_code}: {resp.text[:200]}")
                    return []

                self.model = model_name
                data = resp.json()

                # Safe chained .get() with explicit None checks
                choices = data.get("choices")
                if not choices or len(choices) == 0:
                    logger.info("Perplexity response missing 'choices' field")
                    return []

                message = choices[0].get("message")
                if not message:
                    logger.info("Perplexity response missing 'message' field")
                    return []

                content = message.get("content", "")

                try:
                    clean_content = content.replace("```json", "").replace("```", "").strip()
                    items = json.loads(clean_content)
                    results = []
                    if isinstance(items, list):
                        for item in items:
                            ts = datetime.now(UTC)
                            if "date" in item and item["date"]:
                                try:
                                    ts = datetime.fromisoformat(str(item["date"]).replace("Z", "+00:00"))
                                except Exception as e:
                                    logger.info(f"News search failed: {e}")
                            results.append(
                                {
                                    "title": item.get("title", ""),
                                    "url": item.get("url", ""),
                                    "snippet": item.get("snippet", ""),
                                    "published_at": ts,
                                }
                            )
                        return results
                except json.JSONDecodeError as e:
                    logger.info(f"News search failed: {e}")

                return self._parse_results_fallback(content, num_results)
            except APIBillingError:
                raise
            except Exception as exc:
                logger.info(f"Perplexity request failed for model {model_name}: {exc}")
                continue

        logger.info("Perplexity query failed for all fallback models.")
        return []

    def _parse_results_fallback(self, content: str, num_results: int) -> list[dict[str, Any]]:
        results = []
        lines = content.splitlines()
        for line in lines:
            if not line.strip():
                continue

            md_link = re.search(r"\[(.*?)\]\((https?://.*?)\)", line)
            if md_link:
                title = md_link.group(1)
                url = md_link.group(2)
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": line,  # Use whole line as snippet
                        "published_at": datetime.now(UTC),
                    }
                )
                continue

            if "http" in line:
                parts = line.split("http")
                if len(parts) < 2:
                    continue
                title = parts[0].lstrip("0123456789.-* ").strip()
                # Extract URL safely - take everything up to first space, or entire string if no space
                url_part = parts[1].split(" ")[0] if " " in parts[1] else parts[1]
                url = "http" + url_part
                results.append(
                    {
                        "title": title or "News Headline",
                        "url": url,
                        "snippet": line,
                        "published_at": datetime.now(UTC),
                    }
                )

            if len(results) >= num_results:
                break

        return results
