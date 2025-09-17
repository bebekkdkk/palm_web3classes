import os
from dotenv import load_dotenv
import psycopg2
import psycopg2.extras


class DatabaseConnection:
	"""Simple PostgreSQL connection helper with dict cursor and safe execute."""

	def __init__(self):
		# Load .env from project root (and environment). No code-level defaults.
		load_dotenv()
		missing = [
			name for name in ["PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD"]
			if not os.getenv(name)
		]
		if missing:
			raise RuntimeError(
				f"Missing required DB environment variables: {', '.join(missing)}. "
				"Create a .env file with PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD."
			)

		self.host = os.getenv("PGHOST")
		port_str = os.getenv("PGPORT")
		try:
			self.port = int(port_str)
		except Exception as e:
			raise ValueError(f"Invalid PGPORT value '{port_str}': must be an integer") from e
		self.database = os.getenv("PGDATABASE")
		self.user = os.getenv("PGUSER")
		self.password = os.getenv("PGPASSWORD")

	def _get_conn(self):
		return psycopg2.connect(
			host=self.host,
			port=self.port,
			dbname=self.database,
			user=self.user,
			password=self.password,
		)

	def execute_query(self, query, params=None, fetch=False, fetchone=False):
		"""
		Execute a query.
		- fetch=True returns list of dict rows
		- fetchone=True returns single dict row or None
		Otherwise returns rowcount.
		"""
		conn = None
		cur = None
		try:
			conn = self._get_conn()
			cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
			cur.execute(query, params or ())
			result = None
			if fetchone:
				result = cur.fetchone()
			elif fetch:
				result = cur.fetchall()
			else:
				result = cur.rowcount
			conn.commit()
			return result
		except Exception as e:
			if conn:
				conn.rollback()
			print(f"DB error: {e}")
			raise
		finally:
			if cur:
				cur.close()
			if conn:
				conn.close()
