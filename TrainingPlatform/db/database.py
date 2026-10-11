# -*- coding: utf-8 -*-
"""
Database Manager for AI Vision Training Platform
Using SQLite for local persistent storage.
"""
import os
import sqlite3
import json
from datetime import datetime


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self.close()


class DatabaseManager:
    def __init__(self, db_path=None):
        if db_path is None:
            from TrainingPlatform.core.env_manager import get_workspace_root
            workspace = get_workspace_root()
            data_dir = os.path.join(workspace, "runs")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "platform.db")

            # 兼容性迁移：若 runs/platform.db 尚不存在，但旧的 ~/.ai_vision_platform/platform.db 存在，自动平滑迁移
            old_db = os.path.join(os.path.expanduser("~"), ".ai_vision_platform", "platform.db")
            if not os.path.exists(db_path) and os.path.exists(old_db):
                try:
                    import shutil
                    shutil.copy2(old_db, db_path)
                except Exception:
                    pass
        self.db_path = db_path
        self._init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path, factory=_ClosingConnection)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Projects
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                task_type TEXT NOT NULL, -- detect, segment, pose, classify, obb
                create_time TEXT NOT NULL,
                update_time TEXT NOT NULL
            )
            """)

            # 2. Datasets
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                format TEXT DEFAULT 'yolo',
                image_count INTEGER DEFAULT 0,
                class_count INTEGER DEFAULT 0,
                classes_json TEXT DEFAULT '[]',
                status TEXT DEFAULT 'VALID',
                create_time TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE SET NULL
            )
            """)

            # 3. Training Tasks
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT NOT NULL,
                project_id INTEGER NOT NULL,
                dataset_id INTEGER NOT NULL,
                model_name TEXT NOT NULL, -- YOLOv8n, YOLO11n, etc.
                config_json TEXT NOT NULL, -- Hyperparameters
                status TEXT DEFAULT 'WAITING', -- WAITING, RUNNING, SUCCESS, FAILED, STOPPED
                current_epoch INTEGER DEFAULT 0,
                total_epochs INTEGER DEFAULT 100,
                progress REAL DEFAULT 0.0,
                best_map50 REAL DEFAULT 0.0,
                best_map50_95 REAL DEFAULT 0.0,
                save_dir TEXT,
                log_path TEXT,
                start_time TEXT,
                end_time TEXT,
                FOREIGN KEY (project_id) REFERENCES projects (id),
                FOREIGN KEY (dataset_id) REFERENCES datasets (id)
            )
            """)

            # 4. Models
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                task_id INTEGER,
                project_id INTEGER,
                model_arch TEXT NOT NULL,
                best_pt_path TEXT NOT NULL,
                last_pt_path TEXT,
                metrics_json TEXT,
                export_status TEXT DEFAULT 'PT',
                create_time TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES training_tasks (id) ON DELETE SET NULL,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE SET NULL
            )
            """)
            if "metadata_json" not in {row[1] for row in cursor.execute("PRAGMA table_info(models)")}:
                cursor.execute("ALTER TABLE models ADD COLUMN metadata_json TEXT")

            # 5. Metrics History
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                train_box_loss REAL DEFAULT 0.0,
                train_cls_loss REAL DEFAULT 0.0,
                train_dfl_loss REAL DEFAULT 0.0,
                val_box_loss REAL DEFAULT 0.0,
                val_cls_loss REAL DEFAULT 0.0,
                val_dfl_loss REAL DEFAULT 0.0,
                precision REAL DEFAULT 0.0,
                recall REAL DEFAULT 0.0,
                map50 REAL DEFAULT 0.0,
                map50_95 REAL DEFAULT 0.0,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (task_id) REFERENCES training_tasks (id) ON DELETE CASCADE
            )
            """)
            conn.commit()

    # --- Project APIs ---
    def add_project(self, name, description, task_type="mixed"):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO projects (name, description, task_type, create_time, update_time)
                VALUES (?, ?, ?, ?, ?)
            """, (name, description, task_type, now, now))
            conn.commit()
            return cursor.lastrowid

    def get_projects(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_project_by_id(self, project_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_project(self, project_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM training_tasks WHERE project_id = ?", (project_id,))
            task_ids = [row[0] for row in cursor.fetchall()]
            for tid in task_ids:
                cursor.execute("DELETE FROM metrics_history WHERE task_id = ?", (tid,))
            cursor.execute("DELETE FROM training_tasks WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM models WHERE project_id = ?", (project_id,))
            cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()

    def get_project_summary(self, project_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM training_tasks WHERE project_id = ?", (project_id,))
            task_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM models WHERE project_id = ?", (project_id,))
            model_count = cursor.fetchone()[0]
            return {
                "task_count": task_count,
                "model_count": model_count
            }

    # --- Dataset APIs ---
    def add_dataset(self, project_id, name, path, fmt='yolo', image_count=0, class_count=0, classes=None):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        classes_str = json.dumps(classes or [])
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO datasets (project_id, name, path, format, image_count, class_count, classes_json, create_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (project_id, name, path, fmt, image_count, class_count, classes_str, now))
            conn.commit()
            return cursor.lastrowid

    def get_datasets(self, project_id=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if project_id:
                cursor.execute("SELECT * FROM datasets WHERE project_id = ? ORDER BY id DESC", (project_id,))
            else:
                cursor.execute("SELECT * FROM datasets ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_dataset_by_id(self, dataset_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_dataset(self, dataset_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            conn.commit()

    # --- Task APIs ---
    def add_task(self, task_name, project_id, dataset_id, model_name, config_dict):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_tasks (task_name, project_id, dataset_id, model_name, config_json, total_epochs)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (task_name, project_id, dataset_id, model_name, json.dumps(config_dict), config_dict.get('epochs', 100)))
            conn.commit()
            return cursor.lastrowid

    def update_task_status(self, task_id, status, **kwargs):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            fields = ["status = ?"]
            vals = [status]
            if status == 'RUNNING' and 'start_time' not in kwargs:
                kwargs['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elif status in ('SUCCESS', 'FAILED', 'STOPPED') and 'end_time' not in kwargs:
                kwargs['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for k, v in kwargs.items():
                fields.append(f"{k} = ?")
                vals.append(v)
            vals.append(task_id)
            query = f"UPDATE training_tasks SET {', '.join(fields)} WHERE id = ?"
            cursor.execute(query, vals)
            conn.commit()

    def get_tasks(self, project_id=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if project_id:
                cursor.execute("""
                    SELECT t.*, p.name as project_name, d.name as dataset_name 
                    FROM training_tasks t
                    LEFT JOIN projects p ON t.project_id = p.id
                    LEFT JOIN datasets d ON t.dataset_id = d.id
                    WHERE t.project_id = ? ORDER BY t.id DESC
                """, (project_id,))
            else:
                cursor.execute("""
                    SELECT t.*, p.name as project_name, d.name as dataset_name 
                    FROM training_tasks t
                    LEFT JOIN projects p ON t.project_id = p.id
                    LEFT JOIN datasets d ON t.dataset_id = d.id
                    ORDER BY t.id DESC
                """)
            return [dict(row) for row in cursor.fetchall()]

    def get_task_by_id(self, task_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM training_tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_task(self, task_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM metrics_history WHERE task_id = ?", (task_id,))
            cursor.execute("DELETE FROM training_tasks WHERE id = ?", (task_id,))
            conn.commit()

    # --- Metrics APIs ---
    def record_metrics(self, task_id, epoch, metrics_dict):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics_history (
                    task_id, epoch, train_box_loss, train_cls_loss, train_dfl_loss,
                    val_box_loss, val_cls_loss, val_dfl_loss, precision, recall, map50, map50_95, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id, epoch,
                metrics_dict.get('train_box_loss', 0.0),
                metrics_dict.get('train_cls_loss', 0.0),
                metrics_dict.get('train_dfl_loss', 0.0),
                metrics_dict.get('val_box_loss', 0.0),
                metrics_dict.get('val_cls_loss', 0.0),
                metrics_dict.get('val_dfl_loss', 0.0),
                metrics_dict.get('precision', 0.0),
                metrics_dict.get('recall', 0.0),
                metrics_dict.get('map50', 0.0),
                metrics_dict.get('map50_95', 0.0),
                now
            ))
            conn.commit()

    def get_task_metrics(self, task_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metrics_history WHERE task_id = ? ORDER BY epoch ASC", (task_id,))
            return [dict(row) for row in cursor.fetchall()]

    # --- Model APIs ---
    def get_model_by_id(self, model_id):
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
            return dict(row) if row else None

    def update_model_metadata(self, model_id, metadata):
        with self.get_connection() as conn:
            conn.execute("UPDATE models SET model_arch = ?, metrics_json = ?, metadata_json = ? WHERE id = ?",
                         (metadata.get("task", "unknown"), json.dumps(metadata.get("metrics", {})),
                          json.dumps(metadata, ensure_ascii=False), model_id))

    def import_model(self, project_id, path, metadata):
        normalized = os.path.normcase(os.path.abspath(path))
        existing = next((m for m in self.get_models(project_id)
                         if os.path.normcase(os.path.abspath(m["best_pt_path"])) == normalized), None)
        if existing:
            model_id = existing["id"]
        else:
            model_id = self.add_model(os.path.splitext(os.path.basename(path))[0], None, project_id,
                                      metadata.get("task", "unknown"), os.path.abspath(path))
        self.update_model_metadata(model_id, metadata)
        return model_id

    def add_model(self, name, task_id, project_id, model_arch, best_pt_path, last_pt_path=None, metrics=None):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metrics_str = json.dumps(metrics or {})
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO models (name, task_id, project_id, model_arch, best_pt_path, last_pt_path, metrics_json, create_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, task_id, project_id, model_arch, best_pt_path, last_pt_path, metrics_str, now))
            conn.commit()
            return cursor.lastrowid

    def get_models(self, project_id=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if project_id:
                cursor.execute("""
                    SELECT m.*, p.name as project_name
                    FROM models m
                    LEFT JOIN projects p ON m.project_id = p.id
                    WHERE m.project_id = ? ORDER BY m.id DESC
                """, (project_id,))
            else:
                cursor.execute("""
                    SELECT m.*, p.name as project_name
                    FROM models m
                    LEFT JOIN projects p ON m.project_id = p.id
                    ORDER BY m.id DESC
                """)
            return [dict(row) for row in cursor.fetchall()]

    def delete_model(self, model_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
            conn.commit()

    # --- Overview Summary Stats ---
    def get_dashboard_stats(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM projects")
            project_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM training_tasks")
            task_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM models")
            model_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM datasets")
            dataset_count = cursor.fetchone()[0]

            return {
                "projects": project_count,
                "tasks": task_count,
                "models": model_count,
                "datasets": dataset_count
            }
