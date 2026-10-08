-- ============================================================
-- CreditGuard feature-engineering views
-- Run: psql $DATABASE_URL -f sql/feature_engineering.sql
--
-- Assumes raw tables are in schema creditguard (see ingest.py).
-- Aggregates bureau, bureau_balance, previous_application,
-- and installment payments onto the main application_train.
-- Target: TARGET (1 = default, 0 = paid) is carried through.
-- ============================================================

CREATE SCHEMA IF NOT EXISTS creditguard;

-- ------------------------------------------------------------------
-- 1. Bureau aggregates
-- ------------------------------------------------------------------
CREATE OR REPLACE VIEW creditguard.v_bureau AS
SELECT
    SK_ID_CURR,
    COUNT(*)                                              AS bureau_count,
    SUM(CASE WHEN CREDIT_ACTIVE <> 'Closed' THEN 1 ELSE 0 END)
                                                          AS bureau_active_cnt,
    SUM(CASE WHEN CREDIT_DAY_OVERDUE > 0 THEN 1 ELSE 0 END)
                                                          AS bureau_overdue_cnt,
    ROUND(AVG(AMT_CREDIT_SUM_DEBT), 2)            AS bureau_avg_debt,
    ROUND(MAX(AMT_CREDIT_SUM_DEBT), 2)            AS bureau_max_debt,
    ROUND(SUM(AMT_CREDIT_SUM_DEBT), 2)            AS bureau_total_debt,
    ROUND(AVG(DAYS_CREDIT), 0)::INT               AS bureau_avg_age_days,
    ROUND(AVG(AMT_CREDIT_SUM_LIMIT), 2)           AS bureau_avg_limit,
    ROUND(SUM(AMT_CREDIT_SUM_OVERDUE), 2)         AS bureau_total_overdue
FROM creditguard.bureau
GROUP BY SK_ID_CURR;

-- ------------------------------------------------------------------
-- 2. Bureau-balance aggregates (monthly snapshots)
-- ------------------------------------------------------------------
CREATE OR REPLACE VIEW creditguard.v_bureau_balance AS
SELECT
    SK_ID_BUREAU,
    COUNT(*)                                              AS bb_count,
    SUM(CASE WHEN STATUS IN ('C', 'X') THEN 1 ELSE 0 END)
                                                          AS bb_paid_cnt,
    SUM(CASE WHEN STATUS = '1' THEN 1 ELSE 0 END)       AS bb_1m_delay,
    SUM(CASE WHEN STATUS IN ('2', '3', '4', '5') THEN 1
        ELSE 0 END)                                        AS bb_heavy_delay,
    ROUND(AVG(STATUS::NUMERIC), 2)                       AS bb_avg_status
FROM creditguard.bureau_balance
WHERE STATUS <> 'C'
GROUP BY SK_ID_BUREAU;

-- ------------------------------------------------------------------
-- 3. Previous-application aggregates
-- ------------------------------------------------------------------
CREATE OR REPLACE VIEW creditguard.v_prev_app AS
SELECT
    SK_ID_CURR,
    COUNT(*)                                              AS prev_app_cnt,
    ROUND(AVG(AMT_CREDIT), 2)                         AS prev_avg_credit,
    ROUND(AVG(AMT_ANNUITY), 2)                        AS prev_avg_annuity,
    ROUND(AVG(AMT_DOWN_PAYMENT), 2)                   AS prev_avg_down_payment,
    ROUND(AVG(DAYS_DECISION), 0)::INT                 AS prev_avg_decision_age,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END)
                                                          AS prev_approved_cnt,
    SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1 ELSE 0 END)
                                                          AS prev_refused_cnt,
    ROUND(AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Approved'
                   THEN AMT_CREDIT / NULLIF(AMT_ANNUITY, 0) END), 2)
                                                          AS prev_avg_installment_ratio
FROM creditguard.previous_application
GROUP BY SK_ID_CURR;

-- ------------------------------------------------------------------
-- 4. Installment-payment aggregates
-- ------------------------------------------------------------------
CREATE OR REPLACE VIEW creditguard.v_installments AS
SELECT
    SK_ID_PREV,
    COUNT(*)                                              AS inst_count,
    ROUND(AVG(INSTALTMENT_AMT_IN_DEFERRED_PERIOD), 2) AS inst_avg_deferred,
    ROUND(AVG(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT), 0)::INT
                                                          AS inst_avg_late_days,
    SUM(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END)
                                                          AS inst_late_cnt,
    ROUND(AVG(AMT_INSTALMENT), 2)                     AS inst_avg_amount,
    ROUND(AVG(AMT_PAYMENT), 2)                          AS inst_avg_payment
FROM creditguard.installments_payment
GROUP BY SK_ID_PREV;

-- ------------------------------------------------------------------
-- 5. Merged feature matrix (final view)
-- ------------------------------------------------------------------
DROP VIEW IF EXISTS creditguard.v_feature_matrix CASCADE;
CREATE OR REPLACE VIEW creditguard.v_feature_matrix AS
SELECT
    -- application base
    a.SK_ID_CURR,
    a.TARGET,
    a.NAME_CONTRACT_TYPE,
    a.CODE_GENDER,
    a.FLAG_OWN_CAR,
    a.FLAG_OWN_REALTY,
    a.CNT_CHILDREN,
    a.AMT_INCOME_TOTAL,
    a.AMT_CREDIT,
    a.AMT_ANNUITY,
    a.AMT_GOODS_PRICE,
    a.NAME_INCOME_TYPE,
    a.NAME_EDUCATION_TYPE,
    a.NAME_FAMILY_STATUS,
    a.NAME_HOUSING_TYPE,
    a.OCCUPATION_TYPE,
    a.DAYS_BIRTH,
    a.DAYS_EMPLOYED,
    a.DAYS_REGISTRATION,
    a.DAYS_ID_PUBLISH,
    a.OWN_CAR_AGE,
    a.REGION_RATING_CLIENT,
    a.REGION_RATING_CLIENT_W_CITY,
    a.REGION_POPULATION_RELATIVE,
    a.EXT_SOURCE_1,
    a.EXT_SOURCE_2,
    a.EXT_SOURCE_3,
    a.APARTMENTS_AVG,
    a.BASEMENTAREA_AVG,
    a.YEARS_BEGINEXPLUATATION_AVG,
    a.YEARS_BUILD_AVG,
    a.COMMONAREA_AVG,
    a.ELEVATORS_AVG,
    a.ENTRANCES_AVG,
    a.FLOORSMAX_AVG,
    a.FLOORSMIN_AVG,
    a.LANDAREA_AVG,
    a.LIVINGAPARTMENTS_AVG,
    a.LIVINGAREA_AVG,
    a.NONLIVINGAREA_AVG,
    a.EMERGENCYSTATE_MODE,
    a.OBS_30_CNT_SOCIAL_CIRCLE,
    a.DEF_30_CNT_SOCIAL_CIRCLE,
    a.OBS_60_CNT_SOCIAL_CIRCLE,
    a.DEF_60_CNT_SOCIAL_CIRCLE,

    -- derived application features
    ROUND(a.AMT_CREDIT / NULLIF(a.AMT_INCOME_TOTAL, 0), 2)
                                                          AS credit_to_income_ratio,
    ROUND(a.AMT_ANNUITY / NULLIF(a.AMT_CREDIT, 0), 4)   AS annuity_to_credit_ratio,
    ROUND(a.AMT_GOODS_PRICE / NULLIF(a.AMT_CREDIT, 0), 4)
                                                          AS goods_to_credit_ratio,
    (EXTRACT(EPOCH FROM CURRENT_DATE) / 86400)::INT
                                                          - (a.DAYS_BIRTH / 365)::INT
                                                          AS applicant_age_years,
    CASE WHEN a.DAYS_EMPLOYED < -365
         THEN ((EXTRACT(EPOCH FROM CURRENT_DATE) / 86400)::INT
               - (a.DAYS_EMPLOYED / 365)::INT)
         ELSE NULL
    END                                                   AS employment_years,

    -- bureau aggregates
    COALESCE(b.bureau_count, 0)                          AS bureau_count,
    COALESCE(b.bureau_active_cnt, 0)                     AS bureau_active_cnt,
    COALESCE(b.bureau_overdue_cnt, 0)                  AS bureau_overdue_cnt,
    COALESCE(b.bureau_avg_debt, 0)                     AS bureau_avg_debt,
    COALESCE(b.bureau_max_debt, 0)                     AS bureau_max_debt,
    COALESCE(b.bureau_total_debt, 0)                   AS bureau_total_debt,
    COALESCE(b.bureau_avg_age_days, 0)                 AS bureau_avg_age_days,
    COALESCE(b.bureau_avg_limit, 0)                    AS bureau_avg_limit,
    COALESCE(b.bureau_total_overdue, 0)                AS bureau_total_overdue,

    -- previous-application aggregates
    COALESCE(pa.prev_app_cnt, 0)                         AS prev_app_cnt,
    COALESCE(pa.prev_avg_credit, 0)                    AS prev_avg_credit,
    COALESCE(pa.prev_avg_annuity, 0)                   AS prev_avg_annuity,
    COALESCE(pa.prev_avg_down_payment, 0)             AS prev_avg_down_payment,
    COALESCE(pa.prev_avg_decision_age, 0)             AS prev_avg_decision_age,
    COALESCE(pa.prev_approved_cnt, 0)                 AS prev_approved_cnt,
    COALESCE(pa.prev_refused_cnt, 0)                  AS prev_refused_cnt,
    COALESCE(pa.prev_avg_installment_ratio, 0)        AS prev_avg_installment_ratio,

    -- installment aggregates (via SK_ID_PREV)
    COALESCE(i.inst_count, 0)                            AS inst_count,
    COALESCE(i.inst_avg_deferred, 0)                   AS inst_avg_deferred,
    COALESCE(i.inst_avg_late_days, 0)                  AS inst_avg_late_days,
    COALESCE(i.inst_late_cnt, 0)                         AS inst_late_cnt,
    COALESCE(i.inst_avg_amount, 0)                       AS inst_avg_amount,
    COALESCE(i.inst_avg_payment, 0)                      AS inst_avg_payment

FROM creditguard.application_train a
LEFT JOIN creditguard.v_bureau b
    ON a.SK_ID_CURR = b.SK_ID_CURR
LEFT JOIN creditguard.v_prev_app pa
    ON a.SK_ID_CURR = pa.SK_ID_CURR
LEFT JOIN creditguard.v_installments i
    ON a.SK_ID_PREV = i.SK_ID_PREV;

SELECT 'feature_engineering complete — view creditguard.v_feature_matrix ready' AS status;