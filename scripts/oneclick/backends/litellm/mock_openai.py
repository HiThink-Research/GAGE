import os
from flask import Flask, request, jsonify

MODEL = os.environ.get('MOCK_MODEL', 'qwen2.5-0.5b-instruct-mlx')
app = Flask(__name__)

@app.route('/v1/chat/completions', methods=['POST'])
@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    payload = request.get_json(force=True, silent=True) or {}
    # 记录请求体，便于确认调用协议
    print(f"[MOCK-OPENAI] {request.path} payload={payload}", flush=True)
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
