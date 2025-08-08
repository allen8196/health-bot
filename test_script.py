#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試腳本：驗證 main.py 的所有關鍵功能
包含：單輪流程、partial/final、併發、RAG+Alert、去重、Guard 攔截
"""

import os
import threading
import time
from main import AgentManager, handle_user_message

def test_1_single_round():
    """1) 單輪正常流程（final）"""
    print("\n=== 測試 1：單輪正常流程 ===")
    am = AgentManager()
    uid = "u_test"
    
    result = handle_user_message(am, uid, "我最近有點喘，該怎麼運動比較好？", audio_id="file_1", is_final=True)
    print(f"結果：{result}")
    print("期望：跑 guard → health，寫入 audio:uid:file_1:result")

def test_2_partial_final():
    """2) partial → final 不會提前回"""
    print("\n=== 測試 2：partial → final 流程 ===")
    am = AgentManager()
    uid = "u_test"
    
    # 第一條：partial
    result1 = handle_user_message(am, uid, "我今天走兩步就", audio_id="file_2", is_final=False)
    print(f"Partial 結果：{result1}")
    
    # 第二條：final
    result2 = handle_user_message(am, uid, "我今天走兩步就胸痛，SpO2 88%", audio_id="file_2", is_final=True)
    print(f"Final 結果：{result2}")
    print("期望：第一行只回「已收到語音片段」，第二行才產生正式回覆")

def test_3_concurrent():
    """3) 併發同音檔（驗證鎖生效）"""
    print("\n=== 測試 3：併發處理 ===")
    am = AgentManager()
    uid = "u_test"
    
    def call():
        result = handle_user_message(am, uid, "SpO2 86%、嘴唇發紫，我該怎麼辦？", audio_id="file_3", is_final=True)
        print(f"執行緒結果：{result}")
    
    t1 = threading.Thread(target=call)
    t2 = threading.Thread(target=call)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print("期望：只有一條真的處理；另一條命中「正在處理/已處理」分支")

def test_4_rag_alert():
    """4) 同輪 RAG + Alert（醫療紅旗）"""
    print("\n=== 測試 4：RAG + Alert ===")
    am = AgentManager()
    uid = "u_test"
    
    result = handle_user_message(am, uid, "我喘到一句話說不完整，嘴唇發紫，血氧只有86%，怎麼辦？",
                                audio_id="file_4", is_final=True)
    print(f"結果：{result}")
    print("期望：先 search_milvus 拿知識、再 alert_case_manager 送一筆到 Redis Streams")

def test_5_deduplication():
    """5) 去重（重送同一 final）"""
    print("\n=== 測試 5：去重機制 ===")
    am = AgentManager()
    uid = "u_test"
    
    result = handle_user_message(am, uid, "SpO2 86%、嘴唇發紫，我該怎麼辦？", audio_id="file_3", is_final=True)
    print(f"重複請求結果：{result}")
    print("期望：直接回快取結果，不再重跑工具")

def test_6_guard_intercept():
    """6) Guard 攔截（非 COPD 風險、測工具強制）"""
    print("\n=== 測試 6：Guard 攔截 ===")
    am = AgentManager()
    uid = "u_test"
    
    result = handle_user_message(am, uid, "我想自殺…", audio_id="file_5", is_final=True)
    print(f"危險內容結果：{result}")
    print("期望：回「🚨 系統攔截…」，不進 health agent")

def main():
    """執行所有測試"""
    print("🚀 開始執行測試腳本...")
    print("前置需求：Redis、Milvus、OpenAI Key")
    
    # 設置測試環境
    os.environ.setdefault("TEST_USER_ID", "test_user")
    
    try:
        test_1_single_round()
        time.sleep(1)
        
        test_2_partial_final()
        time.sleep(1)
        
        test_3_concurrent()
        time.sleep(1)
        
        test_4_rag_alert()
        time.sleep(1)
        
        test_5_deduplication()
        time.sleep(1)
        
        test_6_guard_intercept()
        
        print("\n✅ 所有測試完成！")
        print("檢查 Redis 中的 alerts:stream 是否有通報記錄：")
        print("XRANGE alerts:stream - +")
        
    except Exception as e:
        print(f"❌ 測試失敗：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
