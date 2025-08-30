"""Simple user management utilities.

Currently stores a single demo user with a hashed password. This can be
extended later to use a database or external identity provider.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from werkzeug.security import generate_password_hash, check_password_hash


@dataclass
class User:
    email: str
    password_hash: str

    def verify_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


def _init_demo_user() -> User:
    email = os.getenv('DEMO_USER_EMAIL', 'demo@example.com')
    raw_pw = os.getenv('DEMO_USER_PASSWORD', 'password123')
    return User(email=email, password_hash=generate_password_hash(raw_pw))


DEMO_USER = _init_demo_user()


def authenticate(email: str, password: str) -> bool:
    if email.lower() == DEMO_USER.email.lower() and DEMO_USER.verify_password(password):
        return True
    return False


__all__ = ["authenticate", "DEMO_USER", "User"]
