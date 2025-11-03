import mysql.connector
from mysql.connector import Error
import os
from pathlib import Path

# --- HELPER FUNCTIONS ---

def create_connection(host='localhost', user='root', password='', database=None):
    """Create database connection, optionally to a specific DB."""
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database # Connect to a specific DB if provided
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

# --- [THIS FUNCTION IS NOW FIXED] ---
def execute_sql_file(connection, filepath):
    """
    Executes a .sql file by splitting it by semicolons.
    This is a fallback for when 'multi=True' is not supported.
    """
    cursor = None
    try:
        cursor = connection.cursor()
                
        with open(filepath, 'r') as f:
            sql_script = f.read()

        # Split the script into individual statements
        statements = sql_script.split(';')

        for statement in statements:
            statement = statement.strip()
            # Execute only if the statement is not empty
            # and is not just a comment
            if statement and not statement.startswith('--'):
                cursor.execute(statement) # Run one statement at a time
        
        connection.commit()
        print(f"✅ Successfully executed {filepath.name}")
            
    except Error as e:
        # This will now correctly report any SQL syntax errors
        print(f"❌ Error executing {filepath.name}: {e}")
    finally:
        if cursor:
            cursor.close()

# --- MAIN LOGIC (Unchanged) ---

def initialize_database(host='localhost', user='root', password=''):
    """Initialize complete database"""
    
    SCRIPT_DIR = Path(__file__).resolve().parent 
    SCHEMA_PATH = SCRIPT_DIR / 'schema.sql'
    SEED_PATH = SCRIPT_DIR / 'seed_data.sql'
    DB_NAME = 'sqli_demo'

    # --- 1. Connect to MySQL Server (no DB) ---
    print("Connecting to MySQL server...")
    connection = create_connection(host, user, password)
        
    if not connection:
        print("Failed to connect to server. Aborting.")
        return

    # --- 2. Create the Database Explicitly ---
    try:
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        cursor.close()
        connection.commit()
        print(f"✅ Database '{DB_NAME}' created or already exists.")
    except Error as e:
        print(f"❌ Error creating database: {e}")
        connection.close()
        return
    finally:
        connection.close()
        
    # --- 3. Connect to the 'sqli_demo' Database ---
    print(f"Connecting to '{DB_NAME}' database...")
    connection_with_db = create_connection(host, user, password, database=DB_NAME)
    
    if not connection_with_db:
        print(f"❌ Failed to connect to '{DB_NAME}'. Aborting.")
        return

    # --- 4. Run Schema (Tables) and Seed (Data) ---
    try:
        # Execute schema (creates the tables)
        execute_sql_file(connection_with_db, SCHEMA_PATH)
        
        # Execute seed data
        execute_sql_file(connection_with_db, SEED_PATH)
        
        print("\n🎉 Database initialized successfully!")
        
    except Error as e:
        print(f"❌ Error during table creation or seeding: {e}")
    finally:
        connection_with_db.close()
        print("Connection closed.")


if __name__ == "__main__":
    # Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_USER = os.getenv('DB_USER', 'root')
    
    # !! IMPORTANT !!
    # Make sure this password is correct.
    DB_PASSWORD = os.getenv('DB_PASSWORD', '') # <--- YOUR PASSWORD
        
    initialize_database(DB_HOST, DB_USER, DB_PASSWORD)