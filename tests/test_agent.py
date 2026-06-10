"""Unit tests for the Agent class."""

import json
from unittest.mock import MagicMock, patch

import pytest

from agent import Agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_choice(content=None, finish_reason="stop", tool_calls=None):
    """Return a mock ChatCompletion choice."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []
    message.model_dump.return_value = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in (tool_calls or [])
        ],
    }

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_tool_call(call_id, name, arguments_dict):
    """Return a mock tool call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments_dict)
    return tc


# ---------------------------------------------------------------------------
# Basic initialisation
# ---------------------------------------------------------------------------

class TestAgentInit:
    def test_defaults(self):
        with patch("agent.OpenAI"):
            agent = Agent(model="gpt-4o", instructions="You are helpful.")
        assert agent.model == "gpt-4o"
        assert agent.instructions == "You are helpful."
        assert agent.tools == []
        assert agent.tool_callables == {}
        assert agent.history == []
        assert agent.max_tool_iterations == 10

    def test_custom_params(self):
        with patch("agent.OpenAI"):
            agent = Agent(
                model="gpt-3.5-turbo",
                instructions="You are a coder.",
                tools=[{"type": "function", "function": {"name": "f"}}],
                tool_callables={"f": lambda: None},
                max_tool_iterations=5,
            )
        assert agent.model == "gpt-3.5-turbo"
        assert len(agent.tools) == 1
        assert "f" in agent.tool_callables
        assert agent.max_tool_iterations == 5


# ---------------------------------------------------------------------------
# run() – plain text response
# ---------------------------------------------------------------------------

class TestAgentRun:
    def setup_method(self):
        with patch("agent.OpenAI"):
            self.agent = Agent(model="gpt-4o", instructions="Be helpful.")

    def test_plain_text_response(self):
        response = _make_choice(content="Hello, world!", finish_reason="stop")
        self.agent.client.chat.completions.create = MagicMock(return_value=response)

        result = self.agent.run("Hi")

        assert result == "Hello, world!"
        # History should have user message + assistant message
        assert len(self.agent.history) == 2
        assert self.agent.history[0] == {"role": "user", "content": "Hi"}

    def test_history_accumulates(self):
        resp1 = _make_choice(content="First reply.", finish_reason="stop")
        resp2 = _make_choice(content="Second reply.", finish_reason="stop")
        self.agent.client.chat.completions.create = MagicMock(
            side_effect=[resp1, resp2]
        )

        self.agent.run("Message 1")
        self.agent.run("Message 2")

        # 4 messages: user, asst, user, asst
        assert len(self.agent.history) == 4

    def test_system_message_prepended_to_api_call(self):
        response = _make_choice(content="OK", finish_reason="stop")
        create_mock = MagicMock(return_value=response)
        self.agent.client.chat.completions.create = create_mock

        self.agent.run("test")

        call_messages = create_mock.call_args.kwargs["messages"]
        assert call_messages[0] == {"role": "system", "content": "Be helpful."}


# ---------------------------------------------------------------------------
# run() – tool calling
# ---------------------------------------------------------------------------

class TestAgentToolCalling:
    def setup_method(self):
        self.tool_fn = MagicMock(return_value={"result": 42})
        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "my_tool",
                    "description": "A test tool.",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                    },
                },
            }
        ]
        with patch("agent.OpenAI"):
            self.agent = Agent(
                model="gpt-4o",
                instructions="Use tools.",
                tools=tools_schema,
                tool_callables={"my_tool": self.tool_fn},
            )

    def test_tool_call_invoked_then_final_response(self):
        tc = _make_tool_call("call-1", "my_tool", {"x": 5})
        tool_response = _make_choice(finish_reason="tool_calls", tool_calls=[tc])
        final_response = _make_choice(content="Done!", finish_reason="stop")

        self.agent.client.chat.completions.create = MagicMock(
            side_effect=[tool_response, final_response]
        )

        result = self.agent.run("Call my_tool with x=5")

        assert result == "Done!"
        self.tool_fn.assert_called_once_with(x=5)

    def test_tool_result_appended_to_history(self):
        tc = _make_tool_call("call-2", "my_tool", {"x": 7})
        tool_response = _make_choice(finish_reason="tool_calls", tool_calls=[tc])
        final_response = _make_choice(content="All done.", finish_reason="stop")

        self.agent.client.chat.completions.create = MagicMock(
            side_effect=[tool_response, final_response]
        )

        self.agent.run("Go")

        # History: user, asst (tool_calls), tool result, asst (final)
        tool_result_msg = next(
            m for m in self.agent.history if m.get("role") == "tool"
        )
        assert tool_result_msg["tool_call_id"] == "call-2"
        assert json.loads(tool_result_msg["content"]) == {"result": 42}

    def test_unknown_tool_returns_error(self):
        tc = _make_tool_call("call-3", "unknown_tool", {})
        tool_response = _make_choice(finish_reason="tool_calls", tool_calls=[tc])
        final_response = _make_choice(content="Handled.", finish_reason="stop")

        self.agent.client.chat.completions.create = MagicMock(
            side_effect=[tool_response, final_response]
        )

        self.agent.run("Trigger unknown tool")

        tool_result_msg = next(
            m for m in self.agent.history if m.get("role") == "tool"
        )
        payload = json.loads(tool_result_msg["content"])
        assert "error" in payload
        assert "unknown_tool" in payload["error"].lower()

    def test_tool_exception_captured_as_error(self):
        self.tool_fn.side_effect = ValueError("boom")

        tc = _make_tool_call("call-4", "my_tool", {"x": 1})
        tool_response = _make_choice(finish_reason="tool_calls", tool_calls=[tc])
        final_response = _make_choice(content="Handled.", finish_reason="stop")

        self.agent.client.chat.completions.create = MagicMock(
            side_effect=[tool_response, final_response]
        )

        self.agent.run("Trigger exception")

        tool_result_msg = next(
            m for m in self.agent.history if m.get("role") == "tool"
        )
        payload = json.loads(tool_result_msg["content"])
        assert "error" in payload
        assert "boom" in payload["error"]


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestAgentReset:
    def test_reset_clears_history(self):
        with patch("agent.OpenAI"):
            agent = Agent(model="gpt-4o", instructions="Hi.")

        response = _make_choice(content="Reply", finish_reason="stop")
        agent.client.chat.completions.create = MagicMock(return_value=response)

        agent.run("Hello")
        assert len(agent.history) == 2

        agent.reset()
        assert agent.history == []

    def test_run_after_reset_starts_fresh(self):
        with patch("agent.OpenAI"):
            agent = Agent(model="gpt-4o", instructions="Hi.")

        response = _make_choice(content="Reply", finish_reason="stop")
        agent.client.chat.completions.create = MagicMock(return_value=response)

        agent.run("First message")
        agent.reset()
        agent.run("Second message")

        assert len(agent.history) == 2
        assert agent.history[0]["content"] == "Second message"


# ---------------------------------------------------------------------------
# max_tool_iterations guard
# ---------------------------------------------------------------------------

class TestMaxToolIterations:
    def test_stops_after_max_iterations(self):
        tc = _make_tool_call("call-loop", "my_tool", {})
        looping_response = _make_choice(finish_reason="tool_calls", tool_calls=[tc])

        tool_fn = MagicMock(return_value="ok")
        tools_schema = [
            {
                "type": "function",
                "function": {"name": "my_tool", "description": ".", "parameters": {}},
            }
        ]
        with patch("agent.OpenAI"):
            agent = Agent(
                model="gpt-4o",
                instructions=".",
                tools=tools_schema,
                tool_callables={"my_tool": tool_fn},
                max_tool_iterations=3,
            )

        agent.client.chat.completions.create = MagicMock(
            return_value=looping_response
        )

        # Should not loop forever
        agent.run("Loop")

        assert agent.client.chat.completions.create.call_count == 3
