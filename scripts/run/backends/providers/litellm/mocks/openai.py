import os
import requests
from flask import Flask, request, jsonify

MODEL = os.environ.get('MOCK_MODEL', 'qwen2.5-0.5b-instruct-mlx')
MOCK_TARGET = os.environ.get('MOCK_TARGET', '')
API_KEY = os.environ.get('MOCK_API_KEY', '')
app = Flask(__name__)

@app.route('/v1/chat/completions', methods=['POST'])
@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    payload = request.get_json(force=True, silent=True) or {}
    # NOTE: Log the raw request payload to validate the calling protocol.
    print(f"[MOCK-OPENAI] {request.path} payload={payload}", flush=True)
    if MOCK_TARGET:
        headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
        try:
            r = requests.post(MOCK_TARGET.rstrip('/') + '/chat/completions', json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            return jsonify(r.json())
        except Exception as exc:
            return jsonify({"error": f"proxy_failed: {exc}"}), 500
    messages = payload.get('messages') or []
    content = ''
    if messages:
        last = messages[-1]
        content = last.get('content') if isinstance(last, dict) else str(last)
    text = f"mock({MODEL}): {content}" if content else f"mock({MODEL})"
    return jsonify({
        'id': 'mock-openai',
        'object': 'chat.completion',
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': text},
            'finish_reason': 'stop'
        }],
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 18080)), debug=False)
