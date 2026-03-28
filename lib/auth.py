import bcrypt
from lib.db import get_client

# Roles: admin > user > viewer
# admin  — all tabs + admin panel + delete documents
# user   — upload, search, browse (no delete, no admin)
# viewer — search only


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def login(username: str, password: str) -> dict | None:
    """Return user dict on success, None on failure."""
    try:
        result = (
            get_client()
            .table("users")
            .select("*")
            .eq("username", username)
            .eq("is_active", True)
            .execute()
        )
    except Exception:
        return None
    if not result.data:
        return None
    user = result.data[0]
    if verify_password(password, user["password_hash"]):
        return user
    return None


def get_all_users() -> list[dict]:
    result = (
        get_client()
        .table("users")
        .select("id, username, email, role, is_active, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


def create_user(username: str, email: str, password: str, role: str = "user") -> dict:
    result = (
        get_client()
        .table("users")
        .insert({
            "username": username,
            "email": email,
            "password_hash": hash_password(password),
            "role": role,
            "is_active": True,
        })
        .execute()
    )
    return result.data[0]


def update_user(user_id: int, updates: dict) -> None:
    get_client().table("users").update(updates).eq("id", user_id).execute()


def delete_user(user_id: int) -> None:
    get_client().table("users").delete().eq("id", user_id).execute()
