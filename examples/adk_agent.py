"""A complete Google ADK application using the local gateway."""

import argparse
import asyncio

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from local_llm_gateway.integrations.adk import GatewayLlm


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main(profile):
    agent = LlmAgent(
        name="calculator",
        model=GatewayLlm.from_config(profile=profile),
        instruction="Use add for arithmetic, then give the result.",
        tools=[add],
    )
    runner = InMemoryRunner(agent=agent, app_name="gateway_example")
    session = await runner.session_service.create_session(
        app_name="gateway_example", user_id="local"
    )
    async for event in runner.run_async(
        user_id="local",
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text="What is 17 + 25?")]),
    ):
        if event.is_final_response() and event.content:
            print("".join(part.text or "" for part in event.content.parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="primary")
    asyncio.run(main(parser.parse_args().profile))
