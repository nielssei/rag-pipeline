from datetime import timedelta
from temporalio import workflow
from typing import List

@workflow.defn
class RagPipelineWorkflow:
    @workflow.run
    async def run(self, file_path: str) -> dict:
        # Step 1: Extract text from file
        text = await workflow.execute_activity(
            "extract",
            file_path,
            schedule_to_close_timeout=timedelta(seconds=30)
        )

        # Step 2: Chunk the extracted text
        chunks = await workflow.execute_activity(
            "chunk",
            text,
            schedule_to_close_timeout=timedelta(seconds=30)
        )

        # Step 4: Create dense embeddings
        await workflow.execute_activity(
            "index_dense",
            {
                "chunks": chunks,
                "file_path": file_path,
            },
            schedule_to_close_timeout=timedelta(seconds=30)
        )

        return {
            "file": file_path,
            "status": "ok"
        }
    

@workflow.defn
class RagRetrieveWorkflow:
    @workflow.run
    async def run(self, prompt: str):

        # Step 2: Retrieve Dense
        prompt_dense_vector  = await workflow.execute_activity(
            "retrieve_dense",
            prompt,
            schedule_to_close_timeout=timedelta(seconds=30)
        )

        # Step 3: Query DB with Prompt Vector
        result = await workflow.execute_activity(
            "query",
            {
            "prompt_vector": prompt_dense_vector,
            "collection_name": "documents_dense"
            },
            schedule_to_close_timeout=timedelta(seconds=30)
        )

        # Step 3: Query DB with Prompt Vector
        reranked_result = await workflow.execute_activity(
            "rerank",
            {
            "result": result,
            "prompt": prompt
            },
            schedule_to_close_timeout=timedelta(seconds=30)
        )

        answer = await workflow.execute_activity(
            "generate",
            {
            "reranked_result": reranked_result,
            "prompt": prompt
            },
            schedule_to_close_timeout=timedelta(seconds=30)
        )

        return answer