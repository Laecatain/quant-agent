"""
Gemini API 客户端封装。

职责：
1. 从环境变量读取 API Key，避免硬编码密钥；
2. 调用 Google AI Studio 的文本生成接口；
3. 对 429/5xx/网络错误做指数退避重试；
4. 返回纯文本，供上层 Agent 解析为因子候选 JSON。

环境变量优先级：
- GEMINI_API_KEY
- GOOGLE_API_KEY
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import requests


RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class GeminiConfig:
    """Gemini 调用配置。"""

    model: str = "gemini-1.5-flash"
    temperature: float = 0.8
    max_output_tokens: int = 4096
    timeout_seconds: int = 60
    max_retries: int = 4
    base_retry_delay: float = 1.5


class GeminiClient:
    """
    Google AI Studio Gemini API 简单客户端。

    这里选择 requests 直连 REST API，而不是强依赖特定 SDK，原因是：
    - 便于在干净 Anaconda 环境中快速复现；
    - 依赖更少，问题定位更直接；
    - 对 Agent 闭环来说，我们只需要稳定的文本生成能力。
    """

    def __init__(self, api_key: str | None = None, config: GeminiConfig | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.config = config or GeminiConfig()

        if not self.api_key:
            raise ValueError(
                "未找到 Gemini API Key。请先设置环境变量 GEMINI_API_KEY，"
                "例如 PowerShell: $env:GEMINI_API_KEY='你的密钥'"
            )

    @property
    def endpoint(self) -> str:
        """生成 REST API 端点。"""
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:generateContent"

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        调用模型生成文本。

        Args:
            prompt: 用户任务提示词。
            system_prompt: 系统级约束，例如代码格式、风险边界、量化研究原则。

        Returns:
            str: 模型返回的文本。
        """
        payload = self._build_payload(prompt=prompt, system_prompt=system_prompt)
        params = {"key": self.api_key}

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    params=params,
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )

                if response.status_code in RETRY_STATUS_CODES:
                    raise RuntimeError(f"Gemini API 暂时不可用，HTTP {response.status_code}: {response.text[:500]}")

                response.raise_for_status()
                return self._extract_text(response.json())

            except Exception as exc:  # noqa: BLE001 - 客户端边界需要统一包装网络/HTTP/解析异常。
                last_error = exc
                if attempt >= self.config.max_retries:
                    break

                # 指数退避 + 随机抖动，避免多个请求在同一时间点重试。
                delay = self.config.base_retry_delay * (2**attempt) + random.uniform(0, 0.5)
                time.sleep(delay)

        raise RuntimeError(f"Gemini API 调用失败，已重试 {self.config.max_retries} 次：{last_error}")

    def _build_payload(self, prompt: str, system_prompt: str | None) -> dict[str, Any]:
        """构造 generateContent 请求体。"""
        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }

        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        return payload

    @staticmethod
    def _extract_text(response_json: dict[str, Any]) -> str:
        """从 Gemini 响应 JSON 中提取文本。"""
        try:
            parts = response_json["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"无法解析 Gemini 返回结构：{json.dumps(response_json, ensure_ascii=False)[:1000]}") from exc

        if not text:
            raise ValueError(f"Gemini 返回为空：{json.dumps(response_json, ensure_ascii=False)[:1000]}")

        return text
