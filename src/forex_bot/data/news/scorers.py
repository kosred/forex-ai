import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ...domain.errors import APIBillingError

logger = logging.getLogger(__name__)

try:
    from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover
    OpenAI = None
    AuthenticationError = None
    RateLimitError = None
    APIConnectionError = None


@dataclass(slots=True)
class ScoreResult:
    sentiment: float  # -1..1 (negative..positive)
    confidence: float  # 0..1
    direction: str = "neutral"


def _load_key_from_file(path: str, key_name: str) -> str | None:
    """
    Best-effort load of API key from a local text file.
    Expected format: KEY=VALUE per line.
    """
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists():
            return None
        for line in p.read_text().splitlines():
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key_name:
                return v.strip()
    except Exception:
        return None
    return None


class OpenAIScorer:
    def __init__(self, settings) -> None:
        self.available = False
        self.client = None
        self.model = getattr(settings.news, "openai_model", "gpt-5-nano-2025-08-07")
        key_env = getattr(settings.news, "openai_api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(key_env)
        if not api_key:
            secrets_path = getattr(settings, "secrets_file", "keys.txt")
            api_key = _load_key_from_file(secrets_path, key_env)
        if OpenAI is None or not api_key:
            return
        try:
            self.client = OpenAI(api_key=api_key)
            self.available = True
            logger.info(f"OpenAI Scorer initialized with model: {self.model}")
        except Exception as exc:  # pragma: no cover
            logger.warning(f"OpenAI client init failed: {exc}")
            self.client = None
            self.available = False
        self.max_tokens = int(getattr(settings.news, "openai_max_tokens", 256))
        self._last_parse_log = 0.0

    def score(self, title: str, currencies: list[str]) -> ScoreResult | None:
        if not self.available or self.client is None:
            return None

        system_prompt = (
            "You are a Quantitative Macro Strategist at a top-tier FX Hedge Fund. "
            "Analyze the headline for its impact on Global Liquidity and Yield Differentials. "
            "1. Catalyst: Identify the specific driver (Central Bank, Inflation, Geopolitics). "
            "2. Flow: Is this a Risk-On (Bullish high-yielders like AUD) or Risk-Off (Bullish JPY/USD) event? "
            "3. Yield: How does this affect the 2Y/10Y spread for the target currencies? "
            "4. OUTPUT FORMAT (JSON): "
            "   - 'direction': 'buy' | 'sell' | 'neutral' "
            "   - 'sentiment': Float (-1.0 to 1.0). Be aggressive: use 0.8+ for major surprises. "
            "   - 'confidence': Float (0.0 to 1.0). Use 0.9+ for NFP/CPI/Interest Rate results. "
            "   - 'impact_horizon': 'short' | 'medium' | 'long' "
            "   - 'reasoning': Max 15 words explaining the fundamental shift."
        )

        user_content = {"headline": title, "currencies": currencies, "context": "Intraday to Swing trading horizon."}

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            system_prompt + "\n\nReturn ONLY a JSON object. No markdown, no code fences, no extra text."
                        ),
                    },
                    {"role": "user", "content": json.dumps(user_content)},
                ],
                max_completion_tokens=self.max_tokens,
                response_format={"type": "json_object"},  # Enforce valid JSON
            )

            content = resp.choices[0].message.content
            data = _safe_json_object_loads(content)
            if data is None:
                # Enhanced logging for empty/invalid responses
                finish_reason = getattr(resp.choices[0], "finish_reason", "unknown")
                refusal = getattr(resp.choices[0].message, "refusal", None)
                logger.warning(
                    "OpenAI Empty Content. Finish: %s, Refusal: %s, Content: %r",
                    finish_reason,
                    refusal,
                    content,
                )
                self._log_parse_issue(content)
                return None

            direction = str(data.get("direction", "neutral")).lower()
            # HPC FIX: Strict Boundary Enforcement
            confidence = np.clip(float(data.get("confidence", 0.0)), 0.0, 1.0)
            sentiment = np.clip(float(data.get("sentiment", 0.0)), -1.0, 1.0)

            if sentiment == 0.0:
                if direction == "buy":
                    sentiment = 0.5
                elif direction == "sell":
                    sentiment = -0.5

            return ScoreResult(sentiment=float(sentiment), confidence=float(confidence), direction=direction)

        except AuthenticationError as auth_err:
            logger.warning(f"OpenAI authentication failed; disabling OpenAI scorer: {auth_err}")
            self.available = False
            return None
        except RateLimitError as critical_err:
            raise APIBillingError(f"OpenAI Critical Failure: {critical_err}") from critical_err
        except Exception as exc:
            msg = str(exc)
            if "invalid_api_key" in msg or "401" in msg or "unauthorized" in msg.lower():
                logger.warning("OpenAI auth failed (401). Disabling OpenAI scorer.")
                self.available = False
                return None
            logger.warning(f"OpenAI scoring error: {exc}")
            return None

    def _log_parse_issue(self, content) -> None:
        """
        Keep logs clean in live trading: throttle non-fatal parse issues
        (OpenAI returned non-JSON or unexpected format).
        """
        now = time.monotonic()
        if now - self._last_parse_log < 300:
            return
        self._last_parse_log = now
        preview = _preview_text(content, max_len=180)
        logger.info(f"OpenAI scoring returned non-JSON; skipping. preview={preview}")


def _preview_text(value, *, max_len: int) -> str:
    try:
        text = "" if value is None else (value if isinstance(value, str) else str(value))
    except Exception:
        return "<unprintable>"
    text = text.strip().replace("\r", " ").replace("\n", " ")
    if not text:
        return "<empty>"
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if len(lines) <= 2:
        return ""
    # Drop opening fence line (``` or ```json) and trailing fence if present.
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _safe_json_object_loads(content) -> dict | None:
    """
    Best-effort JSON object parser for OpenAI output.

    Even with `response_format={"type":"json_object"}`, some clients/models may return
    markdown fences or pre/post text; keep runtime robust and warning-free.
    """
    try:
        text = "" if content is None else (content if isinstance(content, str) else str(content))
    except Exception:
        return None
    text = text.strip()
    if not text:
        return None
    text = _strip_code_fences(text)
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    candidate = _extract_first_json_object(text)
    if not candidate:
        return None
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else None
    except Exception:
        return None
