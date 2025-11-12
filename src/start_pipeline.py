import asyncio
import os
from temporalio.client import Client
from workflows import RagPipelineWorkflow

async def main():
    client = await Client.connect("localhost:7233")

    directory = 'documents'
    for filename in os.listdir(directory):
        result = await client.start_workflow(
            RagPipelineWorkflow.run,
            f"{directory}/{filename}",
            id=f"wf-{int(asyncio.get_event_loop().time())}",
            task_queue="hello-world",
        )

        # Wait for result
        value = await result.result()
        print("Result:", value)

if __name__ == "__main__":
    asyncio.run(main())