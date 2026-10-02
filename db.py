
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

DB_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "users.db"
)


def get_db_connection():
    """Returns a SQLite connection with Row factory."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database schema and ensures email authentication fields exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if users table exists
    cursor.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='users';"
    )
    table_exists = cursor.fetchone()

    if not table_exists:
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()

    else:
        # Check columns of existing users table
        cursor.execute("PRAGMA table_info(users);")
        columns = [row["name"] for row in cursor.fetchall()]

        needs_migration = (
            "email" not in columns
            or "password_hash" not in columns
        )

        if needs_migration:
            cursor.execute("""
                CREATE TABLE users_v2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Transfer old users if the old database has
            # username and password columns.
            if "username" in columns and "password" in columns:
                cursor.execute(
                    "SELECT id, username, password FROM users;"
                )
                old_users = cursor.fetchall()

                for user in old_users:
                    password = user["password"]

                    if password.startswith(("scrypt:", "pbkdf2:")):
                        password_hash = password
                    else:
                        password_hash = generate_password_hash(password)

                    user_email = (
                        f"{user['username'].lower()}@medical.local"
                    )
                    user_name = user["username"].capitalize()

                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO users_v2
                        (id, email, password_hash, full_name)
                        VALUES (?, ?, ?, ?);
                        """,
                        (
                            user["id"],
                            user_email,
                            password_hash,
                            user_name
                        )
                    )

            cursor.execute("DROP TABLE users;")
            cursor.execute(
                "ALTER TABLE users_v2 RENAME TO users;"
            )

            conn.commit()

    # Create unique index on email if it does not already exist
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email
        ON users(email);
    """)

    conn.commit()
    conn.close()


def create_user(
    email: str,
    password: str,
    full_name: str = ""
) -> dict:
    """Creates a new user with a hashed password."""

    email_clean = email.strip().lower()
    password_hash = generate_password_hash(password)

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users
            (email, password_hash, full_name)
            VALUES (?, ?, ?);
            """,
            (
                email_clean,
                password_hash,
                full_name.strip()
            )
        )

        conn.commit()

        user_id = cursor.lastrowid

        return {
            "id": user_id,
            "email": email_clean,
            "full_name": full_name.strip()
        }

    finally:
        conn.close()


def get_user_by_email(email: str):
    """Fetches user record by email."""

    email_clean = email.strip().lower()

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT * FROM users WHERE email = ?;",
            (email_clean,)
        )

        user = cursor.fetchone()

        return dict(user) if user else None

    finally:
        conn.close()


def get_user_by_id(user_id: int):
    """Fetches user record by ID."""

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT id, email, full_name, created_at
            FROM users
            WHERE id = ?;
            """,
            (user_id,)
        )

        user = cursor.fetchone()

        return dict(user) if user else None

    finally:
        conn.close()


def verify_user(email: str, password: str):
    """Verifies user credentials.

    Returns the user dictionary on successful login,
    otherwise returns None.
    """

    user = get_user_by_email(email)

    if not user:
        return None

    password_hash = user.get("password_hash")

    if password_hash and check_password_hash(
        password_hash,
        password
    ):
        return user

    return None


if __name__ == "__main__":
    init_db()
    print("Database initialized and verified successfully.")
