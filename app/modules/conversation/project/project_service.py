import logging
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
        branch_name: str,
        user_id: str,
        project_id: str,
        repo_path: str = None,
    ):
        try:
            project = Project(
                id=project_id,
                repo_name=repo_name,
                branch_name=branch_name,
                user_id=user_id,
                repo_path=repo_path,
                status=ProjectStatusEnum.SUBMITTED.value,
            )
            self.db.add(project)
            self.db.commit()
            self.db.refresh(project)
            logger.info(f"Project registered: {project_id}")
            return project.id
        except SQLAlchemyError as e:
            logger.error(f"Failed to register project: {e}", exc_info=True)
            self.db.rollback()
            raise ProjectServiceError("Project registration failed.") from e

    async def check_project_exists(
        self, repo_name: str, branch_name: str, user_id: str, repo_path: str = None
    ) -> bool:
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
            return project is not None
        except SQLAlchemyError as e:
            logger.error(f"DB error during project existence check: {e}")
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