import sys
import os

# Add the root project directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.modules.knowledge.inference_service import InferenceService

from app.modules.tools.ask_knowledge_graph_queries import KnowledgeGraphQueryTool, QueryRequest

# Now re-import the tool after injecting mocks
from app.modules.tools.ask_knowledge_graph_queries import KnowledgeGraphQueryTool
# Test function
async def test_tool():
    sql_db = None
    user_id = "31LxbvDjDtwn011DPV4AiR6n"
    tool = KnowledgeGraphQueryTool(sql_db, user_id)

    queries = ["What does the UserService class do?"]
    project_id = "b659a226-9556-467f-830a-0a27b4eaed86"
    result = await tool.arun(queries, project_id)

    print("Test Result:", result)

# Run test
if __name__ == "__main__":
    asyncio.run(test_tool())
