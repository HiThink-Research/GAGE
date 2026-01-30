import threading

from gage_eval.assets.prompts.renderers import JinjaChatPromptRenderer, JinjaDelimitedChatPromptRenderer, PromptContext


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


def test_delimited_chat_renderer_uses_custom_delimiters() -> None:
    template = (
        "System rules.\n"
        "<<<MSG>>>\n"
        "Example User:\n"
        "Change the greeting.\n"
        "<<<MSG>>>\n"
        "Example Assistant:\n"
        "diff --git a/hello.txt b/hello.txt\n"
        "index 1111111..2222222 100644\n"
        "--- a/hello.txt\n"
        "+++ b/hello.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-hello\n"
        "+hello world\n"
        "<<<SECTION>>>\n"
        "{{ sample.messages[0].content }}\n"
    )
    renderer = JinjaDelimitedChatPromptRenderer(
        template,
        mode="header_body_first_user",
        include_system=True,
        message_delimiter="<<<MSG>>>",
        section_delimiter="<<<SECTION>>>",
    )
    context = PromptContext(sample={"messages": [{"role": "user", "content": "Real task"}]})

    result = renderer.render(context)

    assert result.messages is not None
    assert [msg["role"] for msg in result.messages] == ["system", "user", "assistant", "user"]
    assert "diff --git a/hello.txt b/hello.txt" in result.messages[2]["content"]
    assert result.messages[3]["content"] == "Real task"
