"""User management utilities with role support (SQLite-backed).

Backed by a SQLite DB (`data/app.db`) for demo purposes.
Fields per user:
  - email (str, unique)
  - password_hash (str)
  - role (str): one of ["super_admin", "facility_admin", "staff"]
  - name (str, optional)
  - facility_id (int, optional)
  - assigned_patient_ids (list[int], optional)
  - avatar_url (str, optional)

Switch to a real DB server later if needed; the interface keeps that easy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional
from werkzeug.security import check_password_hash
from db import (
    init_db,
    db_get_user_by_email,
    db_insert_user,
    db_list_users,
    db_delete_user_by_email,
    db_update_user,
)


@dataclass
class User:
    email: str
    password_hash: str
    role: str
    name: str | None = None
    facility_id: int | None = None
    assigned_patient_ids: List[int] | None = None
    avatar_url: str | None = None

    def verify_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


# Ensure DB is initialized (and seeded) on import
init_db()


def _coerce_user(obj: dict) -> User:
    return User(
        email=obj.get('email',''),
        password_hash=obj.get('password_hash',''),
        role=obj.get('role','staff'),
        name=obj.get('name'),
        facility_id=obj.get('facility_id'),
        assigned_patient_ids=obj.get('assigned_patient_ids') or [],
        avatar_url=obj.get('avatar_url'),
    )


def get_user_by_email(email: str) -> Optional[User]:
    obj = db_get_user_by_email(email)
    return _coerce_user(obj) if obj else None


def authenticate(email: str, password: str) -> Optional[User]:
    user = get_user_by_email(email)
    if not user:
        return None
    return user if user.verify_password(password) else None


__all__ = [
    "authenticate",
    "get_user_by_email",
    "User",
    # Re-export DB helpers used by app.py admin endpoints
    "db_insert_user",
    "db_list_users",
    "db_delete_user_by_email",
    "db_update_user",
]
