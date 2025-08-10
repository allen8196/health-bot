#!/usr/bin/env python3
"""
清空 user_memory Collection 的所有資料

使用方法:
python clear_memory_collection.py

選項:
- 清空所有資料
- 清空特定使用者的資料
- 只清空空記錄（text為空的記錄）
"""

import os
from pymilvus import connections, Collection
from dotenv import load_dotenv

load_dotenv()

MEM_COLLECTION = os.getenv("MEM_COLLECTION", "user_memory")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")

def connect_milvus():
    """連接到 Milvus"""
    try:
        connections.connect(alias="default", uri=MILVUS_URI)
        print(f"✅ 已連接到 Milvus: {MILVUS_URI}")
        return True
    except Exception as e:
        print(f"❌ 連接 Milvus 失敗: {e}")
        return False

def check_collection_exists():
    """檢查 Collection 是否存在"""
    try:
        col = Collection(MEM_COLLECTION)
        col.load()
        return col
    except Exception as e:
        print(f"❌ Collection '{MEM_COLLECTION}' 不存在或無法載入: {e}")
        return None

def clear_all_data(col: Collection):
    """清空所有資料"""
    try:
        # 獲取所有記錄的 ID
        all_records = col.query(
            expr="id >= 0",  # 獲取所有記錄
            output_fields=["id"],
            limit=100000
        )
        
        if not all_records:
            print("📭 Collection 已經是空的")
            return
        
        print(f"🔍 找到 {len(all_records)} 筆記錄")
        confirm = input("⚠️  確定要清空所有資料嗎？(y/N): ").strip().lower()
        
        if confirm == 'y':
            # 刪除所有記錄
            all_ids = [r["id"] for r in all_records]
            col.delete(expr=f"id in [{','.join(map(str, all_ids))}]")
            print(f"✅ 已刪除 {len(all_ids)} 筆記錄")
        else:
            print("❌ 取消操作")
            
    except Exception as e:
        print(f"❌ 清空資料失敗: {e}")

def clear_user_data(col: Collection, user_id: str):
    """清空特定使用者的資料"""
    try:
        # 獲取該使用者的所有記錄
        user_records = col.query(
            expr=f'user_id == "{user_id}"',
            output_fields=["id", "text"],
            limit=10000
        )
        
        if not user_records:
            print(f"📭 使用者 '{user_id}' 沒有任何記錄")
            return
        
        print(f"🔍 使用者 '{user_id}' 有 {len(user_records)} 筆記錄")
        for i, record in enumerate(user_records[:5]):  # 顯示前5筆
            text_preview = (record.get("text", "")[:50] + "...") if len(record.get("text", "")) > 50 else record.get("text", "")
            print(f"  {i+1}. ID:{record['id']} - {text_preview}")
        
        if len(user_records) > 5:
            print(f"  ... 還有 {len(user_records)-5} 筆記錄")
        
        confirm = input(f"⚠️  確定要刪除使用者 '{user_id}' 的所有記錄嗎？(y/N): ").strip().lower()
        
        if confirm == 'y':
            user_ids = [r["id"] for r in user_records]
            col.delete(expr=f"id in [{','.join(map(str, user_ids))}]")
            print(f"✅ 已刪除使用者 '{user_id}' 的 {len(user_ids)} 筆記錄")
        else:
            print("❌ 取消操作")
            
    except Exception as e:
        print(f"❌ 刪除使用者資料失敗: {e}")

def clear_empty_records(col: Collection):
    """清空空記錄（text為空的記錄）"""
    try:
        # 獲取 text 為空的記錄
        empty_records = col.query(
            expr='text == ""',
            output_fields=["id", "user_id"],
            limit=10000
        )
        
        if not empty_records:
            print("📭 沒有找到空記錄")
            return
        
        print(f"🔍 找到 {len(empty_records)} 筆空記錄")
        user_counts = {}
        for record in empty_records:
            user_id = record.get("user_id", "unknown")
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        
        print("📊 按使用者分布:")
        for user_id, count in user_counts.items():
            print(f"  {user_id}: {count} 筆")
        
        confirm = input("⚠️  確定要清空所有空記錄嗎？(y/N): ").strip().lower()
        
        if confirm == 'y':
            empty_ids = [r["id"] for r in empty_records]
            col.delete(expr=f"id in [{','.join(map(str, empty_ids))}]")
            print(f"✅ 已刪除 {len(empty_ids)} 筆空記錄")
        else:
            print("❌ 取消操作")
            
    except Exception as e:
        print(f"❌ 清空空記錄失敗: {e}")

def main():
    print("🧹 Memory Collection 清理工具")
    print("=" * 40)
    
    if not connect_milvus():
        return
    
    col = check_collection_exists()
    if not col:
        return
    
    while True:
        print("\n請選擇操作:")
        print("1. 清空所有資料")
        print("2. 清空特定使用者資料")
        print("3. 清空空記錄")
        print("0. 退出")
        
        choice = input("\n請輸入選項 (0-3): ").strip()
        
        if choice == "0":
            print("👋 再見！")
            break
        elif choice == "1":
            clear_all_data(col)
        elif choice == "2":
            user_id = input("請輸入使用者 ID: ").strip()
            if user_id:
                clear_user_data(col, user_id)
            else:
                print("❌ 使用者 ID 不能為空")
        elif choice == "3":
            clear_empty_records(col)
        else:
            print("❌ 無效選項，請重新選擇")

if __name__ == "__main__":
    main()
