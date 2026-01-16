import threading

from gage_eval.assets.prompts.renderers import JinjaChatPromptRenderer, PromptContext


def test_jinja_chat_renderer_prepends_system_message() -> None:
    renderer = JinjaChatPromptRenderer("Use tools.", variables=None)
    context = PromptContext(
        sample={"messages": [{"role": "user", "content": "Hi"}]},
    )

    result = renderer.render(context)

    assert result.messages is not None
    assert result.messages[0]["role"] == "system"
    assert result.messages[0]["content"] == "Use tools."
    assert result.messages[1]["role"] == "user"


def test_jinja_chat_renderer_includes_existing_messages_flag() -> None:
    renderer = JinjaChatPromptRenderer("System only.", include_existing=False)
    context = PromptContext(
        sample={"messages": [{"role": "user", "content": "Ignore"}]},
    )

    result = renderer.render(context)

    assert result.messages is not None
    assert len(result.messages) == 1
    assert result.messages[0]["role"] == "system"


def test_prompt_context_handles_unserializable_payload() -> None:
    lock = threading.Lock()
    context = PromptContext(sample={"id": "demo"}, payload={"lock": lock})

    mapping = context.to_mapping()

    assert mapping["payload"]["lock"] is lock
