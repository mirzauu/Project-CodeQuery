from typing import Dict, List

from sqlalchemy.orm import Session

from app.modules.tools.ask_knowledge_graph_queries import (
    get_ask_knowledge_graph_queries_tool,
)
from app.modules.tools.get_code_file_structure import (
    get_code_file_structure_tool,
    GetCodeFileStructureTool,
)
from app.modules.tools.get_code_from_multiple_node_ids_tool import (
    get_code_from_multiple_node_ids_tool,
    GetCodeFromMultipleNodeIdsTool,
)
from app.modules.tools.get_nodes_from_tags_tool import (
    get_nodes_from_tags_tool,
)
# from app.modules.tools.get_code_from_probable_node_name_tool import (
#     get_code_from_probable_node_name_tool,
# )
from app.modules.tools.tool_schema import ToolInfo, ToolInfoWithParameters



from app.modules.provider.provider_service import ProviderService
from langchain_core.tools import StructuredTool


class ToolService:
    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.user_id = user_id
        self.tools = self._initialize_tools()
        self.file_structure_tool = GetCodeFileStructureTool(db)

    def get_tools(self, tool_names: List[str]) -> List[StructuredTool]:
        """get tools if exists"""
        tools = []
        for tool_name in tool_names:
            if self.tools.get(tool_name) is not None:
                tools.append(self.tools[tool_name])
        return tools

    def _initialize_tools(self) -> Dict[str, StructuredTool]:
        tools = {
     
            "ask_knowledge_graph_queries": get_ask_knowledge_graph_queries_tool(
                self.db, self.user_id
            ),
            "get_code_file_structure": get_code_file_structure_tool(self.db),
            "get_code_from_multiple_node_ids": get_code_from_multiple_node_ids_tool(
                self.db, self.user_id
            ),
            # "get_code_from_probable_node_name": get_code_from_probable_node_name_tool(
            #     self.db, self.user_id
            # ),
            "get_nodes_from_tags": get_nodes_from_tags_tool(self.db, self.user_id),
            
           
        }

        

        return tools

    def list_tools(self) -> List[ToolInfo]:
        return [
            ToolInfo(
                id=tool_id,
                name=tool.name,
                description=tool.description,
            )
            for tool_id, tool in self.tools.items()
        ]

    def list_tools_with_parameters(self) -> Dict[str, ToolInfoWithParameters]:
        return {
            tool_id: ToolInfoWithParameters(
                id=tool_id,
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_schema.schema(),
            )
            for tool_id, tool in self.tools.items()
        }
