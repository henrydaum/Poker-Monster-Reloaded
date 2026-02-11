import sqlite3
import uuid
import logging
from dataclasses import dataclass
from collections import Counter

from poker_monster.engine import PHASE_AWAITING_INPUT

logger = logging.getLogger(__name__)

@dataclass
class StepInfo:
    src_id: str = None
    action_id: str = None
    dst_id: str = None
    # Natural language:
    src_text: str = ""
    action_text: str = ""
    dst_text: str = ""

class KnowledgeGraph:
    def __init__(self, db_path="graph.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.init_db()

        self.current_episode_id = None
        self.current_sequence_id = None
        self.current_sequence_start_phase = None
        self.step_counter = 0

    def init_db(self):
        cur = self.conn.cursor()

        # EPISODES - large
        cur.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                outcome REAL DEFAULT 0.0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # SEQUENCES - medium
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sequences (
                sequence_id TEXT PRIMARY KEY,
                episode_id TEXT,
                signature TEXT,
                FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
            )
        """)

        # STEPS - small
        cur.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT,
                sequence_id TEXT,
                step_num INTEGER,
                src_id TEXT,
                action_id TEXT,
                dst_id TEXT,
                src_text TEXT,
                action_text TEXT,
                dst_text TEXT,
                FOREIGN KEY(episode_id) REFERENCES episodes(episode_id),
                FOREIGN KEY(sequence_id) REFERENCES sequences(sequence_id),
                FOREIGN KEY(src_id) REFERENCES nodes(id),
                FOREIGN KEY(dst_id) REFERENCES nodes(id)
            )
        """)

        # NODES - atomic
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                total_reward REAL DEFAULT 0.0,
                count INTEGER DEFAULT 0,
                avg_reward REAL DEFAULT 0.0
            )
        """)

        # EDGES - atomic
        cur.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                src_id TEXT,
                action_id TEXT,
                dst_id TEXT,
                total_reward REAL DEFAULT 0.0,
                count INTEGER DEFAULT 0,
                avg_reward REAL DEFAULT 0.0,
                FOREIGN KEY(src_id) REFERENCES nodes(id),
                PRIMARY KEY (src_id, dst_id, action_id),
                FOREIGN KEY(dst_id) REFERENCES nodes(id)
            )
        """)

        # SKILLS
        cur.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                embedding BLOB,
                preconditions TEXT,
                execution TEXT
            )
        """)

        self.conn.commit()
    
    def start_new_episode(self):
        self.current_episode_id = str(uuid.uuid4())
        self.step_counter = 0
        self.conn.execute("""
            INSERT INTO episodes (episode_id) 
            VALUES (?)
        """, (self.current_episode_id, ))
        self.conn.commit()

    def start_new_sequence(self, start_phase):
        self.current_sequence_id = str(uuid.uuid4())
        self.current_sequence_start_phase = start_phase
        self.conn.execute("""
            INSERT INTO sequences (sequence_id, episode_id)
            VALUES (?, ?)
        """, (self.current_sequence_id, self.current_episode_id))
        self.conn.commit()
    
    def record_step(self, step: StepInfo, boundaries):
        self.create_node(step.src_id)
        self.create_edge(step.src_id, step.action_id, step.dst_id)
        self.create_node(step.dst_id)

        if self.current_sequence_id is None:
            self.start_new_sequence(step.src_id)

        self.conn.execute("""
            INSERT INTO steps (episode_id, sequence_id, step_num, src_id, action_id, dst_id, src_text, action_text, dst_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (self.current_episode_id, self.current_sequence_id, self.step_counter, step.src_id, step.action_id, step.dst_id, step.src_text, step.action_text, step.dst_text))
        
        self.step_counter += 1

        if self.current_sequence_start_phase == PHASE_AWAITING_INPUT:
            # Normal behavior: Close at any boundary
            if step.dst_id in boundaries:
                self.finalize_sequence()
        else:
            # Special behavior: Close boundary at any *other* boundary
            if step.dst_id != self.current_sequence_start_phase:
                self.finalize_sequence()

        self.conn.commit()

    def create_node(self, node_id):
        self.conn.execute("""
            INSERT OR IGNORE INTO nodes (id)
            VALUES (?)
        """, (node_id, ))

    def create_edge(self, src_id, action_id, dst_id):
        self.conn.execute("""
            INSERT OR IGNORE INTO edges (src_id, action_id, dst_id)
            VALUES (?, ?, ?)
        """, (src_id, action_id, dst_id))
    
    def finalize_episode(self, reward: float):
        self.conn.execute("""
            UPDATE episodes
            SET outcome = ?
            WHERE episode_id = ?
        """, (reward, self.current_episode_id))

        steps = self.conn.execute("""
            SELECT src_id, action_id, dst_id 
            FROM steps 
            WHERE episode_id = ?
        """, (self.current_episode_id,)).fetchall()

        if not steps:
            self.conn.commit()
            return

        node_counts = Counter(row['src_id'] for row in steps)
        final_destination = steps[-1]['dst_id']
        if final_destination: 
            node_counts[final_destination] += 1
        
        edge_counts = Counter(
            (row['src_id'], row['action_id'], row['dst_id']) 
            for row in steps
        )

        for node_id, visits in node_counts.items():
            total_added_reward = visits * reward
            self.conn.execute("""
                UPDATE nodes
                SET count = count + ?,
                    total_reward = total_reward + ?,
                    avg_reward = (total_reward + ?) / (count + ?)
                WHERE id = ?
            """, (visits, total_added_reward, total_added_reward, visits, node_id))

        for (src, action, dst), visits in edge_counts.items():
            total_added_reward = visits * reward
            self.conn.execute("""
                UPDATE edges
                SET count = count + ?,
                    total_reward = total_reward + ?,
                    avg_reward = (total_reward + ?) / (count + ?)
                WHERE src_id = ? AND action_id = ? AND dst_id = ?
            """, (visits, total_added_reward, total_added_reward, visits, src, action, dst))

        self.conn.commit()

    def finalize_sequence(self):
        """Compute and store the signature for the current sub-sequence."""
        from thinker import signature_hash

        steps = self.conn.execute("""
            SELECT src_id, action_id, dst_id FROM steps
            WHERE sequence_id = ?
            ORDER BY step_num ASC
        """, (self.current_sequence_id,)).fetchall()

        sig = tuple(s["action_id"] for s in steps)
        sig_hash = signature_hash(sig)

        self.conn.execute("""
            UPDATE sequences SET signature = ?
            WHERE sequence_id = ?
        """, (sig_hash, self.current_sequence_id))

        self.current_sequence_id = None

    def get_sequence_stats(self, signature: str) -> dict | None:
        """Get historical win/loss stats for sequences matching this action pattern."""
        rows = self.conn.execute("""
            SELECT e.outcome
            FROM sequences s
            JOIN episodes e ON s.episode_id = e.episode_id
            WHERE s.signature = ?
        """, (signature,)).fetchall()

        if not rows:
            return None

        outcomes = [r["outcome"] for r in rows]
        wins = sum(1 for o in outcomes if o > 0)
        losses = sum(1 for o in outcomes if o < 0)
        avg = sum(outcomes) / len(outcomes)

        return {
            "wins": wins,
            "losses": losses,
            "avg_reward": round(avg, 3),
            "total": len(outcomes),
        }