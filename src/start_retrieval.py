import argparse
import asyncio
import json
from temporalio.client import Client
from workflows import RagRetrieveWorkflow

async def main(prompt: str):
    client = await Client.connect("localhost:7233")

    result = await client.start_workflow(
        RagRetrieveWorkflow.run,
        prompt,
        id=f"wf-{int(asyncio.get_event_loop().time())}",
        task_queue="hello-world",
    )

    # Wait for result
    value = await result.result()
    print("Result:", json.dumps(value, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="User query prompt")
    args = parser.parse_args()

    asyncio.run(main(args.prompt))