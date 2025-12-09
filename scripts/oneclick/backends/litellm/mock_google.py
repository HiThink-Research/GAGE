import os
import json
from flask import Flask, request, jsonify
import requests

MOCK_TARGET = os.environ.get('MOCK_TARGET', '')
MODEL = os.environ.get('MOCK_MODEL', 'qwen2.5-0.5b-instruct-mlx')
API_KEY = os.environ.get('MOCK_API_KEY', '')
INLINE_ONLY = os.environ.get('MOCK_INLINE_RESPONSE', '0') == '1'

app = Flask(__name__)

@app.route('/v1/models/<path:model_id>:generateContent', methods=['POST'])
@app.route('/v1beta/models/<path:model_id>:generateContent', methods=['POST'])
def gemini_generate(model_id):
    payload = request.get_json(force=True, silent=True) or {}
    print(f"[MOCK-GOOGLE] {request.path} model_id={model_id} payload={payload}", flush=True)
    contents = payload.get('contents') or []
    text_parts = []
    for c in contents:
        parts = c.get('parts') or []
        for p in parts:
            if 'text' in p:
                text_parts.append(p['text'])
    prompt = '\n'.join(text_parts)
    oai_payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
    }
    params = payload.get('generationConfig') or {}
    if params.get('maxOutputTokens') is not None:
        oai_payload['max_tokens'] = params['maxOutputTokens']
    if params.get('temperature') is not None:
        oai_payload['temperature'] = params['temperature']
    text = ''
    if INLINE_ONLY or not MOCK_TARGET:
        text = prompt
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
        'candidates': [{
            'index': 0,
            'content': {'role': 'model', 'parts': [{'text': text}]},
            'finishReason': 'STOP',
<<<<<<< HEAD
        }],
        # litellm 的 Gemini 适配需要 usageMetadata 字段
        'usageMetadata': {'promptTokenCount': 1, 'candidatesTokenCount': 1, 'totalTokenCount': 2},
=======
        }]
>>>>>>> e9d82f1f2ce7b2049f4aa9e13201d15ec18405cb
    })

@app.route('/v1/chat/completions', methods=['POST'])
@app.route('/chat/completions', methods=['POST'])
def openai_compat():
    payload = request.get_json(force=True, silent=True) or {}
    print(f"[MOCK-GOOGLE-OAI] {request.path} payload={payload}", flush=True)
    # 如果指定了下游 OPENAI 兼容服务，则转发；否则直接回显
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
    text = f"mock-google({MODEL}): {content}" if content else f"mock-google({MODEL})"
    return jsonify({
        'id': 'mock-google-openai',
        'object': 'chat.completion',
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': text},
            'finish_reason': 'stop'
        }],
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 18082)), debug=False)
