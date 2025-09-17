import os
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from db_connection import DatabaseConnection

APP_TZ = os.environ.get('APP_TZ', 'Asia/Jakarta')

def now_local():
    try:
        return datetime.now(ZoneInfo(APP_TZ))
    except Exception:
        return datetime.now()

class UserAuthHandler:
    def __init__(self):
        self.db = DatabaseConnection()
        self.users_table = 'users'

    def _hash_password(self, password):
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username, password, group='user'):
        """Register a new user into PostgreSQL users table.
        Assumes table columns: user_id (serial), username, group, password
        """
        try:
            # Check if username exists
            existing = self.db.execute_query(
                f"SELECT user_id FROM {self.users_table} WHERE username=%s",
                (username,),
                fetchone=True
            )
            if existing:
                return False

            hashed_password = self._hash_password(password)
            # Insert
            self.db.execute_query(
                f"INSERT INTO {self.users_table} (username, \"group\", password) VALUES (%s, %s, %s)",
                (username, group, hashed_password)
            )
            return True
            
        except Exception as e:
            print(f"Error in register_user: {str(e)}")
            return False

    def login_user(self, username, password):
        """Login user and return user data if successful from PostgreSQL."""
        try:
            hashed_password = self._hash_password(password)
            row = self.db.execute_query(
                f"SELECT user_id, username, \"group\" FROM {self.users_table} WHERE username=%s AND password=%s",
                (username, hashed_password),
                fetchone=True
            )
            if row:
                return {
                    'user_id': row.get('user_id'),
                    'username': row.get('username'),
                    'group': row.get('group')
                }
            return None
            
        except Exception as e:
            print(f"Error in login_user: {str(e)}")
            return None

    def get_user(self, username):
        """Get user data by username from PostgreSQL"""
        try:
            row = self.db.execute_query(
                f"SELECT user_id, username, \"group\" FROM {self.users_table} WHERE username=%s",
                (username,),
                fetchone=True
            )
            if row:
                return {
                    'user_id': row.get('user_id'),
                    'username': row.get('username'),
                    'group': row.get('group')
                }
            return None
        except Exception as e:
            print(f"Error in get_user: {str(e)}")
            return None

