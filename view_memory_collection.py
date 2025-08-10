#!/usr/bin/env python3
"""
查看 user_memory Collection 的內容

使用方法:
python view_memory_collection.py

功能:
- 查看所有記錄統計
- 查看特定使用者的記錄
- 搜索相似記錄
- 匯出資料
"""

import os
import json
import time
from datetime import datetime
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

def format_timestamp(timestamp):
    """格式化時間戳"""
    try:
        if timestamp:
            dt = datetime.fromtimestamp(timestamp / 1000)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        pass
    return "未知時間"

def show_collection_stats(col: Collection):
    """顯示 Collection 統計資訊"""
    try:
        print("\n📊 Collection 統計資訊")
        print("=" * 50)
        
        # 獲取總記錄數
        all_records = col.query(
            expr="id >= 0",
            output_fields=["id", "user_id", "text"],
            limit=100000
        )
        
        total_count = len(all_records)
        print(f"📝 總記錄數: {total_count}")
        
        if total_count == 0:
            print("📭 Collection 是空的")
            return
        
        # 統計各使用者記錄數
        user_stats = {}
        empty_count = 0
        
        for record in all_records:
            user_id = record.get("user_id", "unknown")
            text = record.get("text", "")
            
            if user_id not in user_stats:
                user_stats[user_id] = {"total": 0, "empty": 0, "with_content": 0}
            
            user_stats[user_id]["total"] += 1
            
            if not text.strip():
                user_stats[user_id]["empty"] += 1
                empty_count += 1
            else:
                user_stats[user_id]["with_content"] += 1
        
        print(f"👥 使用者數量: {len(user_stats)}")
        print(f"📄 有內容記錄: {total_count - empty_count}")
        print(f"📭 空記錄: {empty_count}")
        
        print("\n👥 各使用者統計:")
        for user_id, stats in user_stats.items():
            print(f"  {user_id}: 總共{stats['total']}筆 (有內容:{stats['with_content']}, 空記錄:{stats['empty']})")
        
    except Exception as e:
        print(f"❌ 獲取統計資訊失敗: {e}")

def view_user_records(col: Collection, user_id: str, limit: int = 20):
    """查看特定使用者的記錄"""
    try:
        print(f"\n🔍 使用者 '{user_id}' 的記錄")
        print("=" * 50)
        
        records = col.query(
            expr=f'user_id == "{user_id}"',
            output_fields=["id", "updated_at", "text"],
            limit=limit
        )
        
        if not records:
            print(f"📭 使用者 '{user_id}' 沒有任何記錄")
            return
        
        print(f"📝 找到 {len(records)} 筆記錄 (最多顯示 {limit} 筆)")
        print()
        
        # 按時間排序（最新的在前）
        records.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        
        for i, record in enumerate(records, 1):
            record_id = record.get("id", "unknown")
            timestamp = record.get("updated_at", 0)
            text = record.get("text", "")
            
            formatted_time = format_timestamp(timestamp)
            
            print(f"📄 記錄 #{i}")
            print(f"   ID: {record_id}")
            print(f"   時間: {formatted_time}")
            print(f"   內容: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"   長度: {len(text)} 字元")
            print()
        
    except Exception as e:
        print(f"❌ 查看使用者記錄失敗: {e}")

def search_similar_records(col: Collection, query_text: str, user_id: str = None):
    """搜索相似記錄"""
    try:
        # 這裡需要導入 embedding 函數
        from embedding import safe_to_vector
        
        print(f"\n🔍 搜索與 '{query_text[:50]}...' 相似的記錄")
        print("=" * 50)
        
        # 向量化查詢文本
        query_vector = safe_to_vector(query_text)
        if not query_vector:
            print("❌ 無法向量化查詢文本")
            return
        
        # 構建搜索表達式
        expr = f'user_id == "{user_id}"' if user_id else "id >= 0"
        
        # 執行向量搜索
        results = col.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=10,
            expr=expr,
            output_fields=["user_id", "updated_at", "text"]
        )
        
        if not results or not results[0]:
            print("📭 沒有找到相似記錄")
            return
        
        print(f"📝 找到 {len(results[0])} 筆相似記錄:")
        print()
        
        for i, hit in enumerate(results[0], 1):
            user_id_result = hit.entity.get("user_id", "unknown")
            timestamp = hit.entity.get("updated_at", 0)
            text = hit.entity.get("text", "")
            score = getattr(hit, "score", 0.0)
            
            formatted_time = format_timestamp(timestamp)
            
            print(f"📄 相似記錄 #{i}")
            print(f"   使用者: {user_id_result}")
            print(f"   相似度: {score:.4f}")
            print(f"   時間: {formatted_time}")
            print(f"   內容: {text[:100]}{'...' if len(text) > 100 else ''}")
            print()
        
    except Exception as e:
        print(f"❌ 搜索失敗: {e}")

def export_data(col: Collection, filename: str = None):
    """匯出資料到 JSON 檔案"""
    try:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memory_export_{timestamp}.json"
        
        print(f"\n📤 匯出資料到 '{filename}'")
        
        # 獲取所有記錄
        records = col.query(
            expr="id >= 0",
            output_fields=["id", "user_id", "updated_at", "text"],
            limit=100000
        )
        
        if not records:
            print("📭 沒有資料可匯出")
            return
        
        # 處理資料格式
        export_data = []
        for record in records:
            export_data.append({
                "id": record.get("id"),
                "user_id": record.get("user_id"),
                "updated_at": record.get("updated_at"),
                "formatted_time": format_timestamp(record.get("updated_at")),
                "text": record.get("text"),
                "text_length": len(record.get("text", ""))
            })
        
        # 寫入檔案
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已匯出 {len(export_data)} 筆記錄到 '{filename}'")
        
    except Exception as e:
        print(f"❌ 匯出失敗: {e}")

def main():
    print("👁️  Memory Collection 查看工具")
    print("=" * 40)
    
    if not connect_milvus():
        return
    
    col = check_collection_exists()
    if not col:
        return
    
    while True:
        print("\n請選擇操作:")
        print("1. 查看統計資訊")
        print("2. 查看特定使用者記錄")
        print("3. 搜索相似記錄")
        print("4. 匯出所有資料")
        print("0. 退出")
        
        choice = input("\n請輸入選項 (0-4): ").strip()
        
        if choice == "0":
            print("👋 再見！")
            break
        elif choice == "1":
            show_collection_stats(col)
        elif choice == "2":
            user_id = input("請輸入使用者 ID: ").strip()
            if user_id:
                limit = input("請輸入顯示筆數限制 (預設20): ").strip()
                limit = int(limit) if limit.isdigit() else 20
                view_user_records(col, user_id, limit)
            else:
                print("❌ 使用者 ID 不能為空")
        elif choice == "3":
            query_text = input("請輸入搜索文本: ").strip()
            if query_text:
                user_id = input("請輸入使用者 ID (留空搜索所有使用者): ").strip()
                search_similar_records(col, query_text, user_id if user_id else None)
            else:
                print("❌ 搜索文本不能為空")
        elif choice == "4":
            filename = input("請輸入檔案名稱 (留空使用預設): ").strip()
            export_data(col, filename if filename else None)
        else:
            print("❌ 無效選項，請重新選擇")

if __name__ == "__main__":
    main()
