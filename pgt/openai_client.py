# pgt/openai_client.py
import os
import json
from pathlib import Path

import httpx
from openai import OpenAI

def get_openai_client() -> OpenAI:
    # 读取 secrets.json（与你项目现有逻辑一致）
    secrets = json.load(open("secrets.json", "r", encoding="utf-8"))
    api_key = os.getenv("OPENAI_API_KEY") or secrets["OPENAI_API_KEY"]
    base_url = os.getenv("OPENAI_BASE_URL") or secrets.get("OPENAI_BASE_URL")

    # Clash Verge mixed port（你当前是 7897；也支持用环境变量覆盖）
    proxy = os.getenv("OPENAI_PROXY") or os.getenv("HTTPS_PROXY") or "http://127.0.0.1:7897"

    timeout = httpx.Timeout(300.0, connect=60.0)
    try:
        http_client = httpx.Client(proxy=proxy, timeout=timeout, http2=False)
    except TypeError:
        http_client = httpx.Client(proxies=proxy, timeout=timeout, http2=False)

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        timeout=300,
        max_retries=2,
    )
