"""Minimal A2A evaluation client for AgentBeats security assessment.

Replaces agentbeats-client when SSE streaming fails.
Sends EvalRequest to green-agent via JSON-RPC, polls for completion.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx

TIMEOUT = 300  # 5 minutes per request


async def get_agent_card(base_url: str) -> dict:
    """Fetch agent card from A2A endpoint."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{base_url}/.well-known/agent-card.json")
        r.raise_for_status()
        return r.json()


async def send_jsonrpc(base_url: str, method: str, params: dict) -> dict:
    """Send JSON-RPC request and return response."""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "id": uuid4().hex,
        "params": params,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.post(
            f"{base_url}/",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        return r.json()


async def send_and_poll(green_url: str, eval_request: dict) -> dict:
    """Send evaluation request and poll for results."""
    # Step 1: Send message to green agent
    msg_text = json.dumps(eval_request)
    params = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": msg_text}],
            "messageId": uuid4().hex,
        }
    }

    print(f"[eval_client] Sending eval request to {green_url}")
    resp = await send_jsonrpc(green_url, "message/send", params)

    if "error" in resp:
        print(f"[eval_client] JSON-RPC error: {resp['error']}")
        # Try non-streaming send
        resp = await send_jsonrpc(green_url, "message/send", params)
        if "error" in resp:
            print(f"[eval_client] Second attempt also failed: {resp['error']}")
            return resp

    result = resp.get("result", {})
    print(f"[eval_client] Got response type: {type(result)}")

    # Extract task ID for polling if needed
    if isinstance(result, dict):
        task_id = result.get("id") or result.get("taskId")
        status = result.get("status", {})
        state = status.get("state", "unknown") if isinstance(status, dict) else "unknown"

        if state in ("completed", "failed"):
            print(f"[eval_client] Task {state} immediately")
            return result

        # Poll for completion
        if task_id and state in ("submitted", "working"):
            print(f"[eval_client] Task {task_id} is {state}, polling...")
            for attempt in range(60):  # Max 5 min polling
                await asyncio.sleep(5)
                poll_resp = await send_jsonrpc(
                    green_url, "tasks/get", {"id": task_id}
                )
                poll_result = poll_resp.get("result", {})
                poll_state = poll_result.get("status", {}).get("state", "unknown")
                print(f"[eval_client] Poll {attempt+1}: {poll_state}")
                if poll_state in ("completed", "failed"):
                    return poll_result

    return result


async def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_client.py <scenario.toml> [output.json]")
        sys.exit(1)

    scenario_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    # Parse TOML
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(scenario_path, "rb") as f:
        scenario = tomllib.load(f)

    green_url = scenario["green_agent"]["endpoint"]
    participants = {}
    role_to_id = {}
    for p in scenario.get("participants", []):
        role = p.get("role")
        endpoint = p.get("endpoint")
        agentbeats_id = p.get("agentbeats_id")
        if role and endpoint:
            participants[role] = endpoint
        if role and agentbeats_id:
            role_to_id[role] = agentbeats_id

    config = scenario.get("config", {})

    eval_request = {"participants": participants, "config": config}

    # Verify agents are up
    print(f"[eval_client] Green agent: {green_url}")
    green_card = await get_agent_card(green_url)
    print(f"[eval_client] Green agent: {green_card.get('name', 'unknown')}")

    # Verify participant agents using localhost (since we run on host)
    for role, endpoint in participants.items():
        # Replace Docker container names with localhost for verification
        verify_url = endpoint
        for name in ["agent", "green-agent", "purple-agent"]:
            if f"//{name}:" in verify_url:
                port = verify_url.split(":")[-1]
                verify_url = f"http://localhost:{port}"
                break
        try:
            card = await get_agent_card(verify_url)
            print(f"[eval_client] {role}: {card.get('name', 'unknown')} at {endpoint} (verified via {verify_url})")
        except Exception as e:
            print(f"[eval_client] {role}: WARN could not verify at {verify_url}: {e}")
            print(f"[eval_client] {role}: will use {endpoint} in Docker network")

    # Run evaluation
    start = time.time()
    result = await send_and_poll(green_url, eval_request)
    elapsed = time.time() - start
    print(f"[eval_client] Evaluation completed in {elapsed:.1f}s")

    # Extract results
    output_data = {"participants": role_to_id, "results": []}

    if isinstance(result, dict):
        # Try to extract artifacts
        artifacts = result.get("artifacts", [])
        for artifact in artifacts:
            parts = artifact.get("parts", [])
            for part in parts:
                if isinstance(part, dict):
                    if "data" in part:
                        output_data["results"].append(part["data"])
                    elif "text" in part:
                        try:
                            output_data["results"].append(json.loads(part["text"]))
                        except json.JSONDecodeError:
                            output_data["results"].append({"text": part["text"]})

        # Also check status message
        status_msg = result.get("status", {}).get("message", {})
        if status_msg:
            parts = status_msg.get("parts", [])
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    try:
                        output_data["results"].append(json.loads(part["text"]))
                    except json.JSONDecodeError:
                        pass

    # If no structured results, save raw
    if not output_data["results"]:
        output_data["raw_result"] = result

    print(f"[eval_client] Results: {json.dumps(output_data, indent=2, default=str)[:2000]}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"[eval_client] Saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
