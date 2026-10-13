"""Run from an app directory containing llm-gateway.yaml."""

import argparse
import asyncio

from local_llm_gateway import Gateway, GenerationRequest, Message


async def main(profile):
    gateway = Gateway.from_config()
    response = await gateway.generate(
        GenerationRequest(
            messages=[Message(role="user", content="Explain an LLM gateway in one sentence.")]
        ),
        profile=profile,
    )
    print(response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="primary")
    asyncio.run(main(parser.parse_args().profile))
