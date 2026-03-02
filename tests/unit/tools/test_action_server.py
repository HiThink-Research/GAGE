from __future__ import annotations

import json
from urllib.request import Request, urlopen

from gage_eval.tools.action_server import ActionQueueServer


def _post_json(url: str, payload: dict[str, str]) -> dict[str, str]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:  # noqa: S310 - local test endpoint
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    return {str(key): str(value) for key, value in parsed.items()}


def test_action_server_enqueues_plain_action_text() -> None:
    server = ActionQueueServer(host="127.0.0.1", port=0)
    server.start()
    try:
        host, port = server._server.server_address[:2]
        response = _post_json(f"http://{host}:{port}/tournament/action", {"action": "3"})

        assert response["status"] == "queued"
        assert server.action_queue.get(timeout=1.0) == "3"
    finally:
        server.stop()


def test_action_server_preserves_player_id_in_action_payload() -> None:
    server = ActionQueueServer(host="127.0.0.1", port=0)
    server.start()
    try:
        host, port = server._server.server_address[:2]
        response = _post_json(
            f"http://{host}:{port}/tournament/action",
            {"action": "2", "player_id": "player_1"},
        )

        assert response["status"] == "queued"
        queued = server.action_queue.get(timeout=1.0)
        parsed = json.loads(queued)
        assert parsed["action"] == "2"
        assert parsed["player_id"] == "player_1"
    finally:
        server.stop()
