-- init.sql (統一初始化腳本)

-- 步驟 1: 啟用 pgvector 擴充套件
-- CREATE EXTENSION IF NOT EXISTS vector;

-- 步驟 2: 建立所有需要的 ENUM 型別
CREATE TYPE gender_enum AS ENUM ('male', 'female');
CREATE TYPE water_intake_enum AS ENUM ('low', 'medium', 'good', 'excellent');
CREATE TYPE exercise_level_enum AS ENUM ('rest', 'light', 'moderate', 'active');
CREATE TYPE smoking_level_enum AS ENUM ('none', 'less', 'normal', 'more');
CREATE TYPE completion_status_enum AS ENUM ('all_completed', 'partially_completed', 'incomplete');

-- 步驟 3: 建立 senior_users 表格 (已整合 Profile 和 last_contact_ts 欄位)
CREATE TABLE senior_users (
    line_user_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    gender gender_enum NOT NULL,
    birth_date DATE NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- 【整合點】直接在此處新增 Profile 相關欄位
    profile_personal_background JSONB,
    profile_health_status JSONB,
    profile_life_events JSONB,
    last_contact_ts TIMESTPTZ
);

COMMENT ON TABLE senior_users IS '儲存高齡長者的基本資料、AI 生成的畫像(Profile)以及最後互動時間';

-- 步驟 4: 建立 daily_health_reports 表格
CREATE TABLE daily_health_reports (
    report_id BIGSERIAL PRIMARY KEY,
    line_user_id TEXT NOT NULL REFERENCES senior_users(line_user_id),
    report_date DATE NOT NULL,
    water_intake water_intake_enum,
    medication_taken BOOLEAN,
    exercise_level exercise_level_enum,
    smoking_level smoking_level_enum,
    completion_status completion_status_enum NOT NULL DEFAULT 'incomplete',
    submitted_at TIMESTPTZ DEFAULT NOW(),
    UNIQUE (line_user_id, report_date)
);

-- 步驟 5: 建立索引
CREATE INDEX ON daily_health_reports (line_user_id);
CREATE INDEX ON daily_health_reports (report_date);

-- 步驟 6: 插入初始測試資料
INSERT INTO senior_users (line_user_id, full_name, gender, birth_date) VALUES
('test_user_1', '王阿嬤', 'female', '1950-05-15'),
('test_user_2', '陳阿公', 'male', '1945-12-20')
ON CONFLICT (line_user_id) DO NOTHING;