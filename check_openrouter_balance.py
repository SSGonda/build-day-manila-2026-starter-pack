#!/usr/bin/env python
"""Check OpenRouter API balance."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    print("Error: OPENROUTER_API_KEY not found in .env")
    exit(1)

response = requests.get(
    "https://openrouter.ai/api/v1/credits",
    headers={"Authorization": f"Bearer {api_key}"},
)

if response.status_code == 200:
    data = response.json()["data"]
    print(f"Total credits: ${data['total_credits']:.2f}")
    print(f"Used:         ${data['total_usage']:.2f}")
    print(f"Remaining:    ${data['total_credits'] - data['total_usage']:.2f}")
else:
    print(f"Error: {response.status_code} - {response.text}")
