#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速測試：驗證修復後的關鍵功能
"""

import os
from main import AgentManager, handle_user_message

def quick_test():
    """快速測試基本功能"""
    print("🚀 快速測試開始...")
    
    # 設置環境
    os.environ["CURRENT_USER_ID"] = "test_user"
    
    am = AgentManager()
    uid = "test_user"
    
    # 測試 1：正常對話
    print("\n1. 測試正常對話...")
    result = handle_user_message(am, uid, "我最近有點喘", audio_id="test_1", is_final=True)
    print(f"結果：{result[:100]}...")
    
    # 測試 2：partial → final
    print("\n2. 測試 partial → final...")
    partial = handle_user_message(am, uid, "我今天", audio_id="test_2", is_final=False)
    print(f"Partial: {partial}")
    final = handle_user_message(am, uid, "我今天胸痛", audio_id="test_2", is_final=True)
    print(f"Final: {final[:100]}...")
    
    # 測試 3：併發（簡化版）
    print("\n3. 測試併發處理...")
    result1 = handle_user_message(am, uid, "SpO2 86%", audio_id="test_3", is_final=True)
    result2 = handle_user_message(am, uid, "SpO2 86%", audio_id="test_3", is_final=True)
    print(f"第一次：{result1[:50]}...")
    print(f"第二次：{result2[:50]}...")
    
    # 測試 4：危險內容
    print("\n4. 測試危險內容攔截...")
    danger = handle_user_message(am, uid, "我想自殺", audio_id="test_4", is_final=True)
    print(f"危險內容：{danger}")
    
    print("\n✅ 快速測試完成！")

if __name__ == "__main__":
    quick_test()
