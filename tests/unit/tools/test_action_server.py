from __future__ import annotations

import json
from queue import Queue
from urllib.error import HTTPError
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


def test_action_server_enqueues_normalized_action_payload() -> None:
    server = ActionQueueServer(host="127.0.0.1", port=0)
    server.start()
    try:
        host, port = server._server.server_address[:2]
        response = _post_json(f"http://{host}:{port}/tournament/action", {"action": "3"})

        assert response["status"] == "queued"
        queued = json.loads(server.action_queue.get(timeout=1.0))
        assert queued["action"] == "3"
        assert queued["move"] == "3"
        assert queued["raw"] == "3"
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


def test_action_server_routes_actions_by_sample_id() -> None:
    server = ActionQueueServer(host="127.0.0.1", port=0)
    queue_a: Queue[str] = Queue()
    queue_b: Queue[str] = Queue()
    server.register_action_queue("sample_a", queue_a)
    server.register_action_queue("sample_b", queue_b)
    server.start()
    try:
        host, port = server._server.server_address[:2]
        response_a = _post_json(
            f"http://{host}:{port}/tournament/action",
            {"action": "A1", "player_id": "Human", "sample_id": "sample_a"},
        )
        response_b = _post_json(
            f"http://{host}:{port}/tournament/action",
            {"action": "B2", "player_id": "Human", "sample_id": "sample_b"},
        )

        assert response_a["status"] == "queued"
        assert response_b["status"] == "queued"
        queued_a = json.loads(queue_a.get(timeout=1.0))
        queued_b = json.loads(queue_b.get(timeout=1.0))
        assert queued_a["sample_id"] == "sample_a"
        assert queued_b["sample_id"] == "sample_b"
        assert queued_a["player_id"] == "Human"
        assert queued_b["player_id"] == "Human"
    finally:
        server.stop()


def test_action_server_requires_sample_id_when_routes_are_registered() -> None:
    server = ActionQueueServer(host="127.0.0.1", port=0)
    server.register_action_queue("sample_a", Queue())
    server.start()
    try:
        host, port = server._server.server_address[:2]
        request = Request(
            f"http://{host}:{port}/tournament/action",
            data=json.dumps({"action": "A1"}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as _:
            raise AssertionError("Expected HTTPError when sample_id is missing")
    except HTTPError as exc:
        assert exc.code == 400
        payload = json.loads(exc.read().decode("utf-8"))
        assert payload["error"] == "missing_sample_id"
    finally:
        server.stop()


def test_action_server_requires_player_id_when_routes_are_registered() -> None:
    server = ActionQueueServer(host="127.0.0.1", port=0)
    server.register_action_queue("sample_a", Queue())
    server.start()
    try:
        host, port = server._server.server_address[:2]
        request = Request(
            f"http://{host}:{port}/tournament/action",
            data=json.dumps(
                {"action": "A1", "sample_id": "sample_a"},
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as _:
            raise AssertionError("Expected HTTPError when player_id is missing")
    except HTTPError as exc:
        assert exc.code == 400
        payload = json.loads(exc.read().decode("utf-8"))
        assert payload["error"] == "missing_player_id"
    finally:
        server.stop()
