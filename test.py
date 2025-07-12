from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# 連線到 Milvus
connections.connect(uri="http://localhost:19530")

collection = Collection("demo1")
collection.load()

# 指定要查看哪些欄位（例如 id、text、embedding）
output_fields = ["id", "text"]  # embedding 可加但太大建議先略過

# 使用 query API 撈全部資料（條件設為空）
results = collection.query(
    expr="id >= 0",  # 不設條件表示撈全部
    output_fields=output_fields
)

# 顯示結果
for r in results:
    print(r)
