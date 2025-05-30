import logging
import uuid
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .project_model import Project
from .project_schema import ProjectStatusEnum

logger = logging.getLogger(__name__)

class ProjectServiceError(Exception):
    """Base exception class for ProjectService errors."""


class ProjectNotFoundError(ProjectServiceError):
    """Raised when a project is not found."""

class ProjectService:
    def __init__(self, db: Session):
        self.db = db

    async def get_project_name(self, project_ids: list) -> str:
        try:
            projects = self.db.query(Project).filter(Project.id.in_(project_ids)).all()
            if not projects:
                raise ProjectNotFoundError("No valid projects found.")
            project_name = projects[0].repo_name
            logger.info(f"Project name: {project_name} retrieved for IDs: {project_ids}")
            return project_name
        except SQLAlchemyError as e:
            logger.error(f"DB error in get_project_name: {e}", exc_info=True)
            raise ProjectServiceError("Failed to retrieve project name.") from e

    async def register_project(
        self,
        repo_name: str,
        user_id: str,
        repo_path: str = None,
    ):
        try:
            project = Project(
                id=str(uuid.uuid4()),
                repo_name=repo_name,
                user_id=user_id,
                repo_path=repo_path,
                status=ProjectStatusEnum.SUBMITTED.value,
            )
            self.db.add(project)
            self.db.commit()
            self.db.refresh(project)
            logger.info(f"Project registered: ")
            return project.id
        except SQLAlchemyError as e:
            logger.error(f"Failed to register project: {e}", exc_info=True)
            self.db.rollback()
            raise ProjectServiceError("Project registration failed.") from e

    async def check_project_exists(
        self, repo_url: str, user_id: str,
    ) -> bool:
        print(repo_url,user_id)
        try:
            project = (
                self.db.query(Project)
                .filter_by(
                    repo_url=repo_url,
                    user_id=user_id,
                   
                )
                .first()
            )
            return project is not None
        except SQLAlchemyError as e:
            logger.exception(f"DB error during project existence check: {e}")
            raise ProjectServiceError("Failed to check project existence.") from e

    async def list_projects(self, user_id: str):
        try:
            projects = self.db.query(Project).filter_by(user_id=user_id).all()
            return [
                {
                    "id": project.id,
                    "repo_name": project.repo_name,
                    "status": project.status,
                }
                for project in projects
            ]
        except SQLAlchemyError as e:
            logger.error(f"Failed to list projects: {e}")
            raise ProjectServiceError("Could not list projects.") from e

    async def get_project_from_db(
        self, repo_name: str, branch_name: str, user_id: str, repo_path: str = None
    ):
        try:
            project = (
                self.db.query(Project)
                .filter_by(
                    repo_name=repo_name,
                    branch_name=branch_name,
                    user_id=user_id,
                    repo_path=repo_path,
                )
                .first()
            )
            return project
        except SQLAlchemyError as e:
            logger.error(f"Error in get_project_from_db: {e}")
            raise ProjectServiceError("Could not retrieve project.") from e

    async def get_project_id(
        self, repo_name: str,  user_id: str, repo_path: str = None
    ) -> str:
        try:
            project_id = (
                self.db.query(Project.id)
                .filter_by(
                    repo_name=repo_name,
                   
                    user_id=user_id,
                    repo_path=repo_path,
                )
                .scalar()
            )
            return project_id
        except SQLAlchemyError as e:
            logger.error(f"Error in get_project_id: {e}")
            raise ProjectServiceError("Could not retrieve project ID.") from e    
        
    async def get_project_from_db_by_repo_url(self, repo_url: str, user_id: str):
        try:
            project = (
                self.db.query(Project)
                .filter_by(repo_url=repo_url, user_id=user_id)
                .first()
            )
            return project
        except SQLAlchemyError as e:
            logger.error(f"Error in get_project_from_db_by_repo_url: {e}")
            raise ProjectServiceError("Could not retrieve project.") from e
        
    async def update_project_status(self, project_id: str, new_status: str):
        """
        Updates the status of a project.

        Args:
            project_id (str): The ID of the project to update.
            new_status (str): The new status to set for the project.

        Raises:
            ProjectNotFoundError: If the project with the given ID is not found.
            ProjectServiceError: If there is a database error during the update.
        """
        try:
            project = self.db.query(Project).filter_by(id=project_id).first()
            if not project:
                raise ProjectNotFoundError(f"Project with ID {project_id} not found.")

            project.status = new_status
            self.db.commit()
            logger.info(f"Updated project {project_id} status to {new_status}.")
        except SQLAlchemyError as e:
            logger.error(f"Failed to update project status: {e}", exc_info=True)
            self.db.rollback()
            raise ProjectServiceError("Failed to update project status.") from e

    def get_project_from_db_by_id_sync(self, project_id: str):
        """
        Retrieves a project by its ID synchronously.

        Args:
            project_id (str): The ID of the project to retrieve.

        Returns:
            Project: The project object if found.

        Raises:
            ProjectNotFoundError: If the project with the given ID is not found.
            ProjectServiceError: If there is a database error during the retrieval.
        """
        try:
            project = self.db.query(Project).filter_by(id=project_id).first()
            if not project:
                raise ProjectNotFoundError(f"Project with ID {project_id} not found.")
            return project
        except SQLAlchemyError as e:
            logger.error(f"Error in get_project_from_db_by_id_sync: {e}", exc_info=True)
            raise ProjectServiceError("Could not retrieve project.") from e