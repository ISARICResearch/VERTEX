from __future__ import annotations # py311 does not need
from sqlalchemy import (
    Boolean,
    Column,
    ForeignKey,
    String,
    Text,
    TIMESTAMP,
    Table,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    declared_attr,
)
from typing import Optional, List
import uuid
import datetime
from flask_security import UserMixin
import secrets


class Base(DeclarativeBase):
    pass

class AuditMixin:
    created_at: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP, default=datetime.datetime.now(tz=datetime.timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP, default=datetime.datetime.now(tz=datetime.timezone.utc),
        onupdate=datetime.datetime.now(tz=datetime.timezone.utc),
        nullable=False
    )

    @declared_attr
    def created_by_id(cls) -> Mapped[Optional[uuid.UUID]]:
        return mapped_column(
            UUID(as_uuid=True),
            ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True
        )

    @declared_attr
    def last_modified_by_id(cls) -> Mapped[Optional[uuid.UUID]]:
        return mapped_column(
            UUID(as_uuid=True),
            ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True
        )

    @declared_attr
    def created_by(cls):
        return relationship("User", foreign_keys=[cls.created_by_id], backref=f"{cls.__name__.lower()}s_created")

    @declared_attr
    def last_modified_by(cls):
        return relationship("User", foreign_keys=[cls.last_modified_by_id], backref=f"{cls.__name__.lower()}s_modified")


# -------------------- Association Table --------------------

user_projects = Table(
    "user_projects",
    Base.metadata,
    Column("user_id", UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("project_id", UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True),
    UniqueConstraint("user_id", "project_id", name="uq_user_project")
)

datetime.datetime.now(tz=datetime.timezone.utc)
# -------------------- Models --------------------

class User(Base, UserMixin):
    __tablename__ = "users"

    def get_id(self):
        return str(self.id)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password: Mapped[str] = mapped_column(Text, nullable=False) # this is hashed but needs this name due to flask-security-too
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Required by Flask-Security-Too 4.x
    fs_uniquifier: Mapped[str] = mapped_column(String(length=255), unique=True, default=lambda: secrets.token_urlsafe(32), nullable=False)
    # required by flask-login
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    created_at: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP, default=datetime.datetime.now(tz=datetime.timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        TIMESTAMP, default=datetime.datetime.now(tz=datetime.timezone.utc), onupdate=datetime.datetime.now(tz=datetime.timezone.utc), nullable=False
    )

    # Projects this user is a member of
    projects: Mapped[List[Project]] = relationship(
        "Project",
        secondary=user_projects,
        back_populates="users"
    )


class Project(AuditMixin, Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    project_dir: Mapped[str] = mapped_column(String, nullable=False)

    # Users assigned to this project
    users: Mapped[List[User]] = relationship(
        "User",
        secondary=user_projects,
        back_populates="projects"
    )
