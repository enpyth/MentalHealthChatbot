import sqlite3
from contextlib import contextmanager
from datetime import datetime
import uuid

@contextmanager
def get_db_connection():
    conn = sqlite3.connect('checkpoints.db')
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Create users table
        cursor.execute('''CREATE TABLE IF NOT EXISTS users
                         (email TEXT PRIMARY KEY, password TEXT)''')
        
        # Create conversations table
        cursor.execute('''CREATE TABLE IF NOT EXISTS conversations
                         (id TEXT PRIMARY KEY,
                          user_email TEXT,
                          title TEXT,
                          created_at TIMESTAMP,
                          last_updated TIMESTAMP,
                          FOREIGN KEY (user_email) REFERENCES users(email))''')
        
        # Create cases table
        cursor.execute('''CREATE TABLE IF NOT EXISTS cases
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          user_id TEXT NOT NULL,
                          name TEXT NOT NULL,
                          email TEXT NOT NULL,
                          gender INTEGER NOT NULL,
                          age INTEGER NOT NULL,
                          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

def get_patient_by_user_id(user_id: str) -> dict | None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        SELECT name, email, gender, age 
        FROM cases 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT 1
        ''', (user_id,))
        
        result = cursor.fetchone()
        if result:
            return {
                'name': result[0],
                'email': result[1],
                'gender': result[2],
                'age': result[3]
            }
        return None

def insert_case(user_id: str, patient_data: dict):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        INSERT INTO cases (user_id, name, email, gender, age)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            patient_data['name'],
            patient_data['email'],
            patient_data['gender'],
            patient_data['age']
        ))
        conn.commit()

# User authentication functions
def create_user(email: str, password: str) -> bool:
    conn = sqlite3.connect('checkpoints.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (email, password) VALUES (?, ?)',
                 (email, hash_password(password)))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

def authenticate_user(email: str, password: str):
    conn = sqlite3.connect('checkpoints.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email = ? AND password = ?', 
              (email, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user

# Conversation management functions
def create_conversation(user_email: str, thread_id: str = None, title: str = "New Conversation") -> str:
    conn = sqlite3.connect('checkpoints.db')
    c = conn.cursor()
    now = datetime.now().isoformat()
    
    if thread_id is None:
        thread_id = str(uuid.uuid4())
    
    try:
        c.execute('''INSERT INTO conversations (id, user_email, title, created_at, last_updated)
                     VALUES (?, ?, ?, ?, ?)''', (thread_id, user_email, title, now, now))
        conn.commit()
    except sqlite3.IntegrityError:
        thread_id = str(uuid.uuid4())
        c.execute('''INSERT INTO conversations (id, user_email, title, created_at, last_updated)
                     VALUES (?, ?, ?, ?, ?)''', (thread_id, user_email, title, now, now))
        conn.commit()
    
    conn.close()
    return thread_id

def get_user_conversations(user_email: str):
    conn = sqlite3.connect('checkpoints.db')
    c = conn.cursor()
    c.execute('''SELECT id, title, created_at, last_updated 
                 FROM conversations 
                 WHERE user_email = ?
                 ORDER BY last_updated DESC''', (user_email,))
    conversations = c.fetchall()
    conn.close()
    return conversations

def update_conversation_timestamp(thread_id: str):
    conn = sqlite3.connect('checkpoints.db')
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute('''UPDATE conversations 
                 SET last_updated = ? 
                 WHERE id = ?''', (now, thread_id))
    conn.commit()
    conn.close()

# Helper functions
def hash_password(password: str) -> str:
    from hashlib import sha256
    return sha256(password.encode()).hexdigest() 