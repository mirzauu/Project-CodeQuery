import logging
import os
import shutil
import traceback
import asyncio
from contextlib import contextmanager
from pathlib import Path
from git import Repo, GitCommandError
from fastapi import HTTPException
from typing import Tuple
from tempfile import TemporaryDirectory
from blar_graph.db_managers import Neo4jManager
from blar_graph.graph_construction.core.graph_builder import GraphConstructor
from fastapi import HTTPException
from git import Repo
from sqlalchemy.orm import Session

from app.core.config import config_provider
from app.modules.knowledge.graph_service import CodeGraphService
from .parsing_helper import (
    ParseHelper,
    ParsingFailedError,
    ParsingServiceError,
)
from app.modules.knowledge.inference_service import InferenceService
from app.modules.conversation.project.project_schema import ProjectStatusEnum
from app.modules.conversation.project.project_service import ProjectService

from app.modules.knowledge.schema.parsing_schema import ParseRequest

logger = logging.getLogger(__name__)


class ParsingService:
    BASE_REPO_DIR = "repos"

    def __init__(self, db: Session, user_id: str):
        self.db = db
        self.parse_helper = ParseHelper(db)
        self.project_service = ProjectService(db)
        self.inference_service = InferenceService(db, user_id)

    @contextmanager
    def change_dir(self, path):
        old_dir = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(old_dir)

    async def parse_validator(self, repo_details: str, user_id: str):
        # 1. Validate the Git repo URL
        await self.is_valid_git_repo_url(repo_details)

        # 2. Check if the project already exists
        project_exists = await self.project_service.check_project_exists(repo_details, user_id)

        if not project_exists:
            # Project does not exist — clone and register
            repo, repo_name, repo_path = await self.clone_or_copy_repository(repo_details, user_id)
            
            # Register the new project
            project_id = await self.project_service.register_project(repo_name, user_id, repo_path)

            # Parse the project
            await self.parse_directory(repo_details, user_id, project_id=project_id)
        else:
            # Project exists — fetch it from DB to get details
            project = await self.project_service.get_project_from_db_by_repo_url(repo_details.repo_link, user_id)

            if not project:
                raise HTTPException(status_code=500, detail="Project exists but could not be retrieved.")

            project_id = project.id
            repo_name = project.repo_name
            repo_path = project.repo_path

            # Parse the existing project
            await self.parse_directory(repo_details, user_id, project_id=project_id)


    async def parse_directory(
        self,
        repo_details: ParseRequest,
        user_id: str,
        project_id: int,
        cleanup_graph: bool = True,
    ):
        project_manager = ProjectService(self.db)
        extracted_dir = None
        try:
            if cleanup_graph:
                neo4j_config = config_provider.get_neo4j_config()

                try:
                    code_graph_service = CodeGraphService(
                        neo4j_config["uri"],
                        neo4j_config["username"],
                        neo4j_config["password"],
                        self.db,
                    )

                    # code_graph_service.cleanup_graph(project_id)
                except Exception as e:
                    logger.error(f"Error in cleanup_graph: {e}")
                    raise HTTPException(status_code=500, detail="Internal server error")

            repo,repo_name,user_repo_path = await self.clone_or_copy_repository(
                repo_details, user_id
            )
            

            if isinstance(repo, Repo):
                language = self.parse_helper.detect_repo_language(user_repo_path)
            else:
                languages = repo.get_languages()
                if languages:
                    language = max(languages, key=languages.get).lower()
                else:
                    language = self.parse_helper.detect_repo_language(user_repo_path)
            print(f"Detected language: {language}")
            await self.analyze_directory(
                user_repo_path, project_id, user_id, self.db, language
            )
            message = "The project has been parsed successfully"
            return {"message": message, "id": project_id}

        except ParsingServiceError as e:
            message = str(f"{project_id} Failed during parsing: " + str(e))
            await project_manager.update_project_status(
                project_id, ProjectStatusEnum.ERROR
            )
 


    def create_neo4j_indices(self, graph_manager):
        # Create existing indices from blar_graph
        graph_manager.create_entityId_index()
        graph_manager.create_node_id_index()
        graph_manager.create_function_name_index()

        with graph_manager.driver.session() as session:
            # Existing composite index for repo_id and node_id
            node_query = """
                CREATE INDEX repo_id_node_id_NODE IF NOT EXISTS FOR (n:NODE) ON (n.repoId, n.node_id)
                """
            session.run(node_query)

            # New composite index for name and repo_id to speed up node name lookups
            name_repo_query = """
                CREATE INDEX node_name_repo_id_NODE IF NOT EXISTS FOR (n:NODE) ON (n.name, n.repoId)
                """
            session.run(name_repo_query)

            # New index for relationship types - using correct Neo4j syntax
            rel_type_query = """
                CREATE LOOKUP INDEX relationship_type_lookup IF NOT EXISTS FOR ()-[r]->() ON EACH type(r)
                """
            session.run(rel_type_query)

    async def analyze_directory(
        self,
        extracted_dir: str,
        project_id: int,
        user_id: str,
        db,
        language: str,
       
    ):
        logger.info(
            f"Parsing project {project_id}: Analyzing directory: {extracted_dir}"
        )
   

        if language in ["python", "javascript", "typescript"]:
            graph_manager = Neo4jManager(project_id, user_id)
            self.create_neo4j_indices(
                graph_manager
            )  # commented since indices are created already

            try:
                graph_constructor = GraphConstructor(user_id, extracted_dir)
                n, r = graph_constructor.build_graph()
                graph_manager.create_nodes(n)
                graph_manager.create_edges(r)
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.PARSED
                )
               

                # Generate docstrings using InferenceService
                await self.inference_service.run_inference(project_id)
                logger.info(f"DEBUGNEO4J: After inference project {project_id}")
                self.inference_service.log_graph_stats(project_id)
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.READY
                )
         
              
            except Exception as e:
                logger.error(e)
                logger.error(traceback.format_exc())
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.ERROR
                )
               
            finally:
                graph_manager.close()
        elif language != "other":
            try:
                neo4j_config = config_provider.get_neo4j_config()
                service = CodeGraphService(
                    neo4j_config["uri"],
                    neo4j_config["username"],
                    neo4j_config["password"],
                    db,
                )

                service.create_and_store_graph(extracted_dir, project_id, user_id)

                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.PARSED
                )
                # Generate docstrings using InferenceService
                await self.inference_service.run_inference(project_id)
                logger.info(f"DEBUGNEO4J: After inference project {project_id}")
                self.inference_service.log_graph_stats(project_id)
                await self.project_service.update_project_status(
                    project_id, ProjectStatusEnum.READY
                )
               
                self.inference_service.log_graph_stats(project_id)
            finally:
                service.close()
                logger.info(f"DEBUGNEO4J: After close service {project_id}")
                self.inference_service.log_graph_stats(project_id)
        else:
            await self.project_service.update_project_status(
                project_id, ProjectStatusEnum.ERROR
            )

            self.inference_service.log_graph_stats(project_id)
            raise ParsingFailedError(
                "Repository doesn't consist of a language currently supported."
            )
    
    async def clone_or_copy_repository(self, repo_details, user_id: str) -> Repo:
        # if not hasattr(repo_details, "repo_url") or not repo_details:
        #     raise HTTPException(status_code=400, detail="Repository URL is required.")

        repo_url = repo_details
        repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        user_repo_path = os.path.join(self.BASE_REPO_DIR, user_id, repo_name)

        Path(user_repo_path).parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(user_repo_path):
            try:
                repo = Repo(user_repo_path)
                if repo.bare:
                    raise HTTPException(status_code=400, detail="Existing repo is corrupted or bare.")
                return repo,repo_name,user_repo_path
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load existing repo: {str(e)}")

        try:
            repo = Repo.clone_from(repo_url, user_repo_path)
            return repo,repo_name,user_repo_path
        except GitCommandError as e:
            raise HTTPException(status_code=500, detail=f"Git clone failed: {str(e)}")    
        
    async def is_valid_git_repo_url(self,repo_url: str) -> bool:
        """
        Asynchronously checks if a given URL is a valid, publicly accessible Git repository.
        """
        def try_clone(temp_dir: str) -> bool:
            try:
                Repo.clone_from(repo_url, temp_dir)
                return True
            except GitCommandError:
                return False

        with TemporaryDirectory() as tmpdir:
            return await asyncio.to_thread(try_clone, tmpdir)    

    # async def duplicate_graph(self, old_repo_id: str, new_repo_id: str):
    #     await self.search_service.clone_search_indices(old_repo_id, new_repo_id)
    #     node_batch_size = 3000  # Fixed batch size for nodes
    #     relationship_batch_size = 3000  # Fixed batch size for relationships
    #     try:
    #         # Step 1: Fetch and duplicate nodes in batches
    #         with self.inference_service.driver.session() as session:
    #             offset = 0
    #             while True:
    #                 nodes_query = """
    #                 MATCH (n:NODE {repoId: $old_repo_id})
    #                 RETURN n.node_id AS node_id, n.text AS text, n.file_path AS file_path,
    #                        n.start_line AS start_line, n.end_line AS end_line, n.name AS name,
    #                        COALESCE(n.docstring, '') AS docstring,
    #                        COALESCE(n.embedding, []) AS embedding,
    #                        labels(n) AS labels
    #                 SKIP $offset LIMIT $limit
    #                 """
    #                 nodes_result = session.run(
    #                     nodes_query,
    #                     old_repo_id=old_repo_id,
    #                     offset=offset,
    #                     limit=node_batch_size,
    #                 )
    #                 nodes = [dict(record) for record in nodes_result]

    #                 if not nodes:
    #                     break

    #                 # Insert nodes under the new repo ID, preserving labels, docstring, and embedding
    #                 create_query = """
    #                 UNWIND $batch AS node
    #                 CALL apoc.create.node(node.labels, {
    #                     repoId: $new_repo_id,
    #                     node_id: node.node_id,
    #                     text: node.text,
    #                     file_path: node.file_path,
    #                     start_line: node.start_line,
    #                     end_line: node.end_line,
    #                     name: node.name,
    #                     docstring: node.docstring,
    #                     embedding: node.embedding
    #                 }) YIELD node AS new_node
    #                 RETURN new_node
    #                 """
    #                 session.run(create_query, new_repo_id=new_repo_id, batch=nodes)
    #                 offset += node_batch_size

    #         # Step 2: Fetch and duplicate relationships in batches
    #         with self.inference_service.driver.session() as session:
    #             offset = 0
    #             while True:
    #                 relationships_query = """
    #                 MATCH (n:NODE {repoId: $old_repo_id})-[r]->(m:NODE)
    #                 RETURN n.node_id AS start_node_id, type(r) AS relationship_type, m.node_id AS end_node_id
    #                 SKIP $offset LIMIT $limit
    #                 """
    #                 relationships_result = session.run(
    #                     relationships_query,
    #                     old_repo_id=old_repo_id,
    #                     offset=offset,
    #                     limit=relationship_batch_size,
    #                 )
    #                 relationships = [dict(record) for record in relationships_result]

    #                 if not relationships:
    #                     break

    #                 relationship_query = """
    #                 UNWIND $batch AS relationship
    #                 MATCH (a:NODE {repoId: $new_repo_id, node_id: relationship.start_node_id}),
    #                       (b:NODE {repoId: $new_repo_id, node_id: relationship.end_node_id})
    #                 CALL apoc.create.relationship(a, relationship.relationship_type, {}, b) YIELD rel
    #                 RETURN rel
    #                 """
    #                 session.run(
    #                     relationship_query, new_repo_id=new_repo_id, batch=relationships
    #                 )
    #                 offset += relationship_batch_size

    #         logger.info(
    #             f"Successfully duplicated graph from {old_repo_id} to {new_repo_id}"
    #         )

    #     except Exception as e:
    #         logger.error(
    #             f"Error duplicating graph from {old_repo_id} to {new_repo_id}: {e}"
    #         )
