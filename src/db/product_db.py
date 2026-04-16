import pymysql
from pymysql.cursors import DictCursor
from functools import lru_cache

from src.utils.config_utils import cfg
from src.entities.schemas import ProductRegisterRequest


class ProductDBManager:
    """商品信息 mysqlDB管理"""

    def __init__(self):
        self.db_config = cfg.DB_CONFIG
        self._init_db_table()
        self._cache = {}  # 简单的字典缓存：{id: product_info_dict}

    def _get_connection(self):
        """获取数据库连接"""
        return pymysql.connect(**self.db_config, cursorclass=DictCursor)

    def _init_db_table(self):
        """建表语句（如果不存在则创建）"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS products (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            product_name VARCHAR(255) NOT NULL,
            big_category ENUM('bagged', 'bottled', 'boxed', 'canned') NOT NULL,
            fine_class VARCHAR(100) NOT NULL,
            unit_price DECIMAL(10,2) NOT NULL DEFAULT 0.00,
            sku VARCHAR(50) UNIQUE,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_sql)
            print(">>> MySQL: products 数据表初始化校验完成。")
        except Exception as e:
            print(f"[数据库错误] 初始化商品表失败: {e}")

    @lru_cache(maxsize=1024)
    def get_product_by_ids(self, ids: int | list) -> list[dict] | None:
        """
        根据 ids 查询商品详细信息
        """
        # 统一转为 list，兼容单个 id
        if isinstance(ids, int):
            id_list = [ids]
        elif isinstance(ids, list):
            id_list = ids
        else:
            raise TypeError(
                f"ids 参数必须是 int 或 list[int]，当前类型: {type(ids).__name__}"
            )

        if not id_list:
            return None

        placeholders = ",".join(["%s"] * len(id_list))
        sql = f"SELECT * FROM products WHERE id IN ({placeholders})"
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, id_list)
                    # 返回字典 {"sku": "...", "product_name": "..."}的list
                    return cursor.fetchall()
        except Exception as e:
            ids_str = str(ids) if isinstance(ids, int) else ", ".join(map(str, ids))
            print(f"[数据库查询错误] ids: [{ids_str}], 错误: {e}")
            return None

    def get_product_by_ids_batch(self, ids: list[int]) -> dict[int, dict]:
        """
        批量查询：
        1. 先从内存缓存找
        2. 缺失的部分一次性去 DB 查
        3. 存回缓存并返回
        返回格式: {id: {sku:..., name:...}} 方便 O(1) 索引
        """
        if not ids:
            return {}

        # 去重，防止一帧内出现多个相同商品导致重复查询
        unique_ids = list(set(ids))

        # 1. 找出缓存中没有的 ID
        missing_ids = [idx for idx in unique_ids if idx not in self._cache]

        if missing_ids:
            # 2. 只有当确实有缺失时才查数据库
            placeholders = ",".join(["%s"] * len(missing_ids))
            sql = f"SELECT * FROM products WHERE id IN ({placeholders})"
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(sql, missing_ids)
                        db_results = cursor.fetchall()
                        # 3. 填入缓存
                        for row in db_results:
                            self._cache[row["id"]] = row
            except Exception as e:
                print(f"[DB Batch Error] {e}")

        # 4. 从缓存中构造本次请求的结果映射
        return {idx: self._cache[idx] for idx in unique_ids if idx in self._cache}

    def get_product_name(self, ids: str) -> str:
        """
        快捷获取商品中文名
        """
        product = self.get_product_by_ids(ids)
        if product:
            return product[0].get("product_name", "未知商品名")
        return "未注册商品"

    def insert_product(self, product_req: ProductRegisterRequest):
        """
        注册/更新 新商品信息，并返回 product_id
        """
        # 使用 ON DUPLICATE KEY UPDATE 防止重复注册报错，存在则更新
        sql = """
        INSERT INTO products (product_name, big_category, fine_class, unit_price, sku)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            product_name = VALUES(product_name),
            big_category = VALUES(big_category),
            fine_class = VALUES(fine_class),
            unit_price = VALUES(unit_price)
        """
        product_id = None

        try:
            unit_price = float(product_req.price) if product_req.price else 0.0

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        sql,
                        (
                            product_req.product_name,
                            product_req.big_category,
                            product_req.sku,
                            unit_price,
                            product_req.sku,
                        ),
                    )

                    # 2. 获取 ID
                    # 如果是插入操作，lastrowid 是有效的
                    if cursor.lastrowid != 0:
                        product_id = cursor.lastrowid
                    else:
                        # 如果是更新操作（lastrowid 为 0），通过 SKU 查找对应 ID
                        cursor.execute(
                            "SELECT id FROM products WHERE sku = %s", (product_req.sku,)
                        )
                        result = cursor.fetchone()
                        if result:
                            product_id = result["id"]
                if product_id is None:
                    raise Exception("无法获取有效的 Product ID")

                conn.commit()
                print(
                    f"商品信息已处理。ID: {product_id}, 名称: {product_req.product_name}"
                )
            return product_id
        except Exception as e:
            conn.rollback()
            print(f"[数据库插入错误] 写入商品失败: {str(e)}")
            return None
