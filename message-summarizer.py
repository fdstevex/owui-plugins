"""
title: Message Summarizer
author: @stevex
author_url: https://github.com/fdstevex
funding_url: https://github.com/fdstevex
version: 0.1
"""

from pydantic import BaseModel, Field
from typing import Optional
import json
import aiohttp
import asyncio


class Filter:
    # Valves: Configuration options for the filter
    class Valves(BaseModel):
        summarize_threshold: int = Field(
            default=10, description="Summarize when there are this many messages"
        )
        summarize_count: int = Field(
            default=5, description="Number of messages to summarize"
        )
        llm_endpoint: str = Field(
            default="http://localhost:11434/v1/chat/completions",
            description="OpenAI compatible completion endpoint",
        )
        llm_api_key: str = Field(default="none", description="API key if required")

        llm_model: str = Field(
            default="hf.co/TheDrummer/Cydonia-24B-v2.1-GGUF:latest",
            description="Model to use for summarization",
        )
        pass

    class UserValves(BaseModel):
        llm_model: str = Field(
            default="hf.co/TheDrummer/Cydonia-24B-v2.1-GGUF:latest",
            description="Model to use for summarization",
        )
        summarize_threshold: int = Field(
            default=10, description="Summarize when there are this many messages"
        )
        summarize_count: int = Field(
            default=5, description="Number of messages to summarize"
        )
        pass

    def __init__(self):
        self.valves = self.Valves()

    async def call_llm(self, prompt) -> str:
        # Function to make an LLM call to do the summarization
        endpoint = self.valves.llm_endpoint
        api_key = self.valves.llm_api_key
        model = self.valves.llm_model

        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": api_key,
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            ) as response:
                response_json = await response.json()
                return response_json["choices"][0]["message"]["content"].strip()

    async def outlet(
        self, body: dict, user: Optional[dict] = None, __event_emitter__=None
    ) -> dict:
        messages = body["messages"]

        if len(messages) < self.valves.summarize_threshold:
            # not time to summarize yet
            return body

        to_summarize = ""
        summarized_count = 0

        try:
            roles = ""
            new_messages = []
            has_system_message = False

            for i, message in enumerate(messages):
                role = message["role"]
                roles = roles + role
                content = message["content"]
                if content is None:
                    continue

                if role == "system" and i == 0:
                    has_system_message = True
                    new_messages.append(message)
                    continue

                if summarized_count >= self.valves.summarize_count:
                    # preserve
                    new_messages.append(message)
                else:
                    # include in the summary
                    summarized_count = summarized_count + 1
                    to_summarize += f"{role}: {content}\n"

            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Summarizing {summarized_count} messages of {len(messages)}",
                        "done": False,
                    },
                }
            )  # type: ignore

            prompt = """
            You are summarizing part of a long conversation. Read the following messages,
            and then write a summary of the significant parts of the conversation to that point.
            ---
            """
            prompt = f"{prompt}{to_summarize}"
            resp = await self.call_llm(prompt)

            # insert the response as an assistant message just after the first system message
            new_messages.insert(
                1 if has_system_message else 0, {"role": "assistant", "content": resp}
            )

            # TODO remove the summarized messages

            body["messages"] = new_messages

            await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Summarization Complete: "
                    + json.dumps(body["messages"]),
                    "done": True,
                },
            }
        )  # type: ignore

        except Exception as e:
            await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": f"Summarization Exception: {str(e)}",
                    "done": True,
                },
            }

          return body
 # type: ignore
