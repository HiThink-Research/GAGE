import os
import json
from flask import Flask, request, jsonify
import requests

MOCK_TARGET = os.environ.get('MOCK_TARGET', '')
MODEL = os.environ.get('MOCK_MODEL', 'qwen2.5-0.5b-instruct-mlx')
API_KEY = os.environ.get('MOCK_API_KEY', '')
INLINE_ONLY = os.environ.get('MOCK_INLINE_RESPONSE', '0') == '1'

app = Flask(__name__)

@app.route('/v1/messages', methods=['POST'])
def anthropic_messages():
    payload = request.get_json(force=True, silent=True) or {}
    print(f"[MOCK-ANTHROPIC] {request.path} payload={payload}", flush=True)
    messages = payload.get('messages') or []
    prompt_parts = []
    for m in messages:
        role = m.get('role')
        content = m.get('content')
        if isinstance(content, list):
            content = ''.join([c.get('text', '') if isinstance(c, dict) else str(c) for c in content])
        prompt_parts.append(f"{role}: {content}")
    # map to OpenAI chat.completions
    oai_payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': '\n'.join(prompt_parts)}],
    }
    if payload.get('max_tokens') is not None:
        oai_payload['max_tokens'] = payload['max_tokens']
    if payload.get('temperature') is not None:
        oai_payload['temperature'] = payload['temperature']
    # call downstream openai-compatible service or inline echo
    text = ''
    if INLINE_ONLY or not MOCK_TARGET:
        text = '\\n'.join(prompt_parts)
    else:
        headers = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
        r = requests.post(MOCK_TARGET.rstrip('/') + '/chat/completions', json=oai_payload, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            choices = data.get('choices') or []
            if choices:
                msg = choices[0].get('message') or choices[0].get('delta') or {}
                text = msg.get('content', '') if isinstance(msg, dict) else str(msg)
    return jsonify({
        'id': 'local-anthropic-mock',
        'type': 'message',
        'model': MODEL,
        'role': 'assistant',
        'content': [{'type': 'text', 'text': text}],
        'stop_reason': 'end_turn',
        'stop_sequence': None,
        'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2},
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 18081)), debug=False)
