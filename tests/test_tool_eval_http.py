import json
import threading
import urllib.request

from needle_timemachine.tool_eval import _make_server


class FakeBackend:
    def complete(self, payload):
        return {"object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}


def test_embedded_chat_completions_endpoint():
    server = _make_server("127.0.0.1", 0, FakeBackend())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = "http://127.0.0.1:%s/v1/chat/completions" % server.server_address[1]
        body = json.dumps({"model": "needle", "messages": [{"role": "user", "content": "hi"}], "tools": []}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as result:
            payload = json.loads(result.read().decode())
        assert payload["object"] == "chat.completion"
    finally:
        server.shutdown()
        server.server_close()
