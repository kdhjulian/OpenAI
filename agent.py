"""OpenAI Agent class with tool-calling support."""

import json
import os
from typing import Any, Callable

from openai import OpenAI


class Agent:
    """A conversational agent backed by OpenAI's Chat Completions API.

    The agent maintains a conversation history and supports tool (function)
    calling so that it can invoke arbitrary Python callables in response to
    model requests.

    Args:
        model: The OpenAI model identifier to use (e.g. ``"gpt-4o"``).
        instructions: A system-level prompt that describes the agent's role
            and behaviour.
        tools: An optional list of tool definitions.  Each entry must be a
            dict that conforms to the OpenAI ``tools`` schema, i.e.::

                {
                    "type": "function",
                    "function": {
                        "name": "my_tool",
                        "description": "...",
                        "parameters": { ... },
                    },
                }

        tool_callables: An optional mapping from tool name to a Python
            callable that will be invoked when the model requests that tool.
            The callable receives the parsed JSON arguments as keyword
            arguments.
        api_key: OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
            environment variable when not provided.
        max_tool_iterations: Maximum number of tool-calling rounds per
            ``run()`` call before returning.  Defaults to ``10``.
    """

    def __init__(
        self,
        model: str,
        instructions: str,
        tools: list[dict] | None = None,
        tool_callables: dict[str, Callable[..., Any]] | None = None,
        api_key: str | None = None,
        max_tool_iterations: int = 10,
    ) -> None:
        self.model = model
        self.instructions = instructions
        self.tools: list[dict] = tools or []
        self.tool_callables: dict[str, Callable[..., Any]] = tool_callables or {}
        self.max_tool_iterations = max_tool_iterations
        self.history: list[dict] = []
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_message: str) -> str:
        """Send *user_message* to the model and return the assistant reply.

        Tool calls requested by the model are executed automatically and
        their results are fed back until the model produces a plain text
        response or ``max_tool_iterations`` is reached.

        Args:
            user_message: The user's message text.

        Returns:
            The final assistant response as a plain string.
        """
        self.history.append({"role": "user", "content": user_message})

        for _ in range(self.max_tool_iterations):
            response = self._call_api()
            choice = response.choices[0]
            message = choice.message

            # Append the raw assistant message (may contain tool_calls)
            self.history.append(message.model_dump(exclude_unset=False))

            if choice.finish_reason == "tool_calls" and message.tool_calls:
                self._handle_tool_calls(message.tool_calls)
                continue

            # Plain text response — we're done
            return message.content or ""

        # Exceeded iteration limit — return whatever the last content was
        last = self.history[-1]
        return last.get("content") or ""

    def reset(self) -> None:
        """Clear the conversation history."""
        self.history.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_api(self):
        """Build the messages list and call the Chat Completions API."""
        messages = [{"role": "system", "content": self.instructions}] + self.history
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs)

    def _handle_tool_calls(self, tool_calls) -> None:
        """Execute each tool call and append results to history."""
        for tool_call in tool_calls:
            name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            result = self._invoke_tool(name, arguments)

            self.history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    def _invoke_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Look up and call the named tool callable.

        Args:
            name: The tool name as declared in ``tool_callables``.
            arguments: Parsed keyword arguments for the callable.

        Returns:
            The return value of the callable, or an error dict when the
            tool is not registered.
        """
        if name not in self.tool_callables:
            return {"error": f"Unknown tool: {name}"}
        try:
            return self.tool_callables[name](**arguments)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
