import asyncio
from temporalio.worker import Worker
from temporalio.client import Client
from workflows import RagPipelineWorkflow, RagRetrieveWorkflow
from activities import extract, chunk, index_dense, retrieve_dense, retrieve_sparse, query, rerank, generate

async def main():
    # Connect to the local Temporal server
    client = await Client.connect("localhost:7233")

    # Create a Worker that listens on a task queue
    worker = Worker(
        client,
        task_queue="hello-world",
        workflows=[RagPipelineWorkflow, RagRetrieveWorkflow],
        activities=[extract, chunk, index_dense, retrieve_dense, retrieve_sparse, query, rerank, generate],
    )

    print("Worker started. Listening on 'hello-world'...")
    await worker.run()

if __name__ == "__main__":
    asyncio.run(main())