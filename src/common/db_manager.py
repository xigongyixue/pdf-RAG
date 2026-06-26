"""MySQL数据库管理模块：存储RAG系统的元数据、摘要和块数据。"""
import json
import pymysql
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class DatabaseManager:
    """MySQL数据库管理器，处理所有数据库操作。"""

    def __init__(self, host: str = "localhost", port: int = 3306, 
                 user: str = "root", password: str = "", 
                 database: str = "pdf_rag", charset: str = "utf8mb4"):
        """初始化数据库连接参数。"""
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.connection = None

    def connect(self):
        """建立数据库连接。"""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                charset=self.charset,
                cursorclass=pymysql.cursors.DictCursor
            )
            # 创建数据库（如果不存在）
            self._create_database_if_not_exists()
            # 选择数据库
            self.connection.select_db(self.database)
            # 创建表（如果不存在）
            self._create_tables_if_not_exists()
            # print(f"已连接到MySQL数据库: {self.database}")
        except Exception as e:
            raise Exception(f"数据库连接失败: {e}")

    def disconnect(self):
        """关闭数据库连接。"""
        if self.connection:
            self.connection.close()
            self.connection = None

    @contextmanager
    def get_cursor(self):
        """获取数据库游标的上下文管理器。"""
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()
            raise e
        finally:
            cursor.close()

    def _create_database_if_not_exists(self):
        """创建数据库（如果不存在）。"""
        with self.get_cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{self.database}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")

    def _create_tables_if_not_exists(self):
        """创建所有必需的表（如果不存在）。"""
        # 1. 文章表
        with self.get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL UNIQUE,
                    chunks_file VARCHAR(255),
                    abstract_file VARCHAR(255),
                    chunk_count INT DEFAULT 0,
                    chunk_start INT DEFAULT 0,
                    chunk_end INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_name (name),
                    INDEX idx_chunk_range (chunk_start, chunk_end)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

        # 2. 摘要表
        with self.get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS abstracts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    article_name VARCHAR(255) NOT NULL,
                    title TEXT,
                    content LONGTEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (article_name) REFERENCES articles(name) ON DELETE CASCADE,
                    INDEX idx_article_name (article_name),
                    FULLTEXT idx_content (title, content)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

        # 3. 块表（全局唯一索引）
        with self.get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    global_index INT NOT NULL UNIQUE,
                    article_name VARCHAR(255) NOT NULL,
                    section VARCHAR(255),
                    content LONGTEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (article_name) REFERENCES articles(name) ON DELETE CASCADE,
                    INDEX idx_global_index (global_index),
                    INDEX idx_article_name (article_name),
                    INDEX idx_section (section),
                    FULLTEXT idx_content (content)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

    # ─── 文章操作 ────────────────────────────
    def add_article(self, name: str, chunks_file: str = None, 
                   abstract_file: str = None, chunk_count: int = 0,
                   chunk_start: int = 0, chunk_end: int = 0) -> int:
        """添加文章记录。返回文章ID。"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO articles (name, chunks_file, abstract_file, chunk_count, chunk_start, chunk_end)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    chunks_file = VALUES(chunks_file),
                    abstract_file = VALUES(abstract_file),
                    chunk_count = VALUES(chunk_count),
                    chunk_start = VALUES(chunk_start),
                    chunk_end = VALUES(chunk_end),
                    updated_at = CURRENT_TIMESTAMP
            """, (name, chunks_file, abstract_file, chunk_count, chunk_start, chunk_end))
            return cursor.lastrowid

    def get_article(self, name: str) -> Optional[Dict[str, Any]]:
        """获取文章信息。"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM articles WHERE name = %s", (name,))
            return cursor.fetchone()

    def get_all_articles(self) -> List[Dict[str, Any]]:
        """获取所有文章信息。"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM articles ORDER BY chunk_start")
            return cursor.fetchall()

    def delete_article(self, name: str) -> bool:
        """删除文章（级联删除相关的摘要和块）。"""
        with self.get_cursor() as cursor:
            cursor.execute("DELETE FROM articles WHERE name = %s", (name,))
            return cursor.rowcount > 0

    def delete_all_articles(self) -> int:
        """删除所有文章（级联删除所有摘要和块）。"""
        with self.get_cursor() as cursor:
            cursor.execute("DELETE FROM articles")
            return cursor.rowcount

    def update_article_chunk_range(self, name: str, chunk_start: int, chunk_end: int, chunk_count: int):
        """更新文章的块范围和数量。"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                UPDATE articles 
                SET chunk_start = %s, chunk_end = %s, chunk_count = %s, updated_at = CURRENT_TIMESTAMP
                WHERE name = %s
            """, (chunk_start, chunk_end, chunk_count, name))

    # ─── 摘要操作 ────────────────────────────
    def add_or_update_abstract(self, article_name: str, title: str, content: str) -> int:
        """添加或更新摘要。返回摘要ID。"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO abstracts (article_name, title, content)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    title = VALUES(title),
                    content = VALUES(content),
                    updated_at = CURRENT_TIMESTAMP
            """, (article_name, title, content))
            return cursor.lastrowid

    def get_abstract(self, article_name: str) -> Optional[Dict[str, Any]]:
        """获取文章摘要。"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM abstracts WHERE article_name = %s", (article_name,))
            return cursor.fetchone()

    def get_all_abstracts(self) -> List[Dict[str, Any]]:
        """获取所有摘要。"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM abstracts ORDER BY article_name")
            return cursor.fetchall()

    # ─── 块操作 ───────────────────────────────
    def add_chunk(self, global_index: int, article_name: str, section: str, content: str) -> int:
        """添加块。返回块ID。如果global_index已存在则更新。"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO chunks (global_index, article_name, section, content)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    article_name = VALUES(article_name),
                    section = VALUES(section),
                    content = VALUES(content),
                    updated_at = CURRENT_TIMESTAMP
            """, (global_index, article_name, section, content))
            return cursor.lastrowid

    def add_chunks_batch(self, chunks: List[Dict[str, Any]]) -> int:
        """批量添加块。返回添加的块数量。"""
        if not chunks:
            return 0
        
        with self.get_cursor() as cursor:
            # 使用INSERT ... ON DUPLICATE KEY UPDATE批量插入
            values = []
            for chunk in chunks:
                values.append((
                    chunk['global_index'],
                    chunk['article_name'],
                    chunk['section'],
                    chunk['content']
                ))
            
            cursor.executemany("""
                INSERT INTO chunks (global_index, article_name, section, content)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    article_name = VALUES(article_name),
                    section = VALUES(section),
                    content = VALUES(content),
                    updated_at = CURRENT_TIMESTAMP
            """, values)
            return cursor.rowcount

    def get_chunk(self, global_index: int) -> Optional[Dict[str, Any]]:
        """根据全局索引获取块。"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT * FROM chunks WHERE global_index = %s", (global_index,))
            return cursor.fetchone()

    def get_chunks_by_article(self, article_name: str) -> List[Dict[str, Any]]:
        """获取文章的所有块。"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM chunks 
                WHERE article_name = %s 
                ORDER BY global_index
            """, (article_name,))
            return cursor.fetchall()

    def get_chunks_in_range(self, start: int, end: int) -> List[Dict[str, Any]]:
        """获取指定范围内的块。"""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM chunks 
                WHERE global_index >= %s AND global_index <= %s 
                ORDER BY global_index
            """, (start, end))
            return cursor.fetchall()

    def delete_chunks_by_article(self, article_name: str) -> int:
        """删除文章的所有块。"""
        with self.get_cursor() as cursor:
            cursor.execute("DELETE FROM chunks WHERE article_name = %s", (article_name,))
            return cursor.rowcount

    def get_max_global_index(self) -> int:
        """获取当前最大的全局块索引。"""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT COALESCE(MAX(global_index), -1) as max_index FROM chunks")
            result = cursor.fetchone()
            return result['max_index'] if result else -1

    # ─── 元数据操作 ──────────────────────────
    def get_meta(self) -> Dict[str, Any]:
        """获取系统元数据。"""
        articles = self.get_all_articles()
        total_chunks = self.get_max_global_index() + 1
        
        return {
            "total_chunks": total_chunks,
            "total_articles": len(articles),
            "articles": [
                {
                    "name": article['name'],
                    "chunks_file": article['chunks_file'],
                    "abstract_file": article['abstract_file'],
                    "chunk_count": article['chunk_count'],
                    "chunk_range": [article['chunk_start'], article['chunk_end']]
                }
                for article in articles
            ]
        }

    # ─── 数据迁移 ──────────────────────────
    def migrate_from_json(self, meta_path: str, chunks_dir: str, abstracts_dir: str = None):
        """从JSON文件迁移数据到MySQL数据库。"""
        import os
        
        # 读取meta.json
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        print(f"开始迁移数据，共 {meta['total_articles']} 篇文章，{meta['total_chunks']} 个块")
        
        # 迁移每篇文章
        for article_info in meta['articles']:
            article_name = article_info['name']
            print(f"迁移文章: {article_name}")
            
            # 1. 添加文章记录
            self.add_article(
                name=article_name,
                chunks_file=article_info['chunks_file'],
                abstract_file=article_info['abstract_file'],
                chunk_count=article_info['chunk_count'],
                chunk_start=article_info['chunk_range'][0],
                chunk_end=article_info['chunk_range'][1]
            )
            
            # 2. 迁移摘要
            abstract_file = article_info['abstract_file']
            if abstracts_dir:
                abstract_path = os.path.join(abstracts_dir, abstract_file)
            else:
                abstract_path = os.path.join(chunks_dir, abstract_file)
            
            if os.path.exists(abstract_path):
                with open(abstract_path, 'r', encoding='utf-8') as f:
                    abstract = json.load(f)
                self.add_or_update_abstract(
                    article_name=article_name,
                    title=abstract['title'],
                    content=abstract['content']
                )
            
            # 3. 迁移块
            chunks_file = article_info['chunks_file']
            chunks_path = os.path.join(chunks_dir, chunks_file)
            
            if os.path.exists(chunks_path):
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                # 批量插入块
                chunks_data = [
                    {
                        'global_index': chunk['index'],
                        'article_name': article_name,
                        'section': chunk['section'],
                        'content': chunk['content']
                    }
                    for chunk in chunks
                ]
                self.add_chunks_batch(chunks_data)
            
            print(f"  完成: {article_info['chunk_count']} 个块")
        
        print("数据迁移完成！")
