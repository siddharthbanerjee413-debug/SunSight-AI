-- ============================================================================
-- SunSight AI — Persistence Layer (MySQL)
-- ============================================================================
-- This schema defines the relational backbone that stores:
--   1. Static tariff & subsidy data per Indian state.
--   2. An audit log of every user query and predicted ROI.
--
-- Usage:
--   mysql -u root -p < schema.sql
-- ============================================================================

CREATE DATABASE IF NOT EXISTS sunsight_ai
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE sunsight_ai;

-- ---------------------------------------------------------------------------
-- Table 1: state_tariffs
-- ---------------------------------------------------------------------------
-- Stores per-state electricity pricing and government subsidy information.
-- This keeps financial constants out of the ML model so tariff changes
-- never require retraining.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS state_tariffs (
    id              INT             AUTO_INCREMENT PRIMARY KEY,
    state_name      VARCHAR(100)    NOT NULL UNIQUE          COMMENT 'Indian state or union territory name',
    electricity_rate DECIMAL(6,2)   NOT NULL                 COMMENT 'Per-unit electricity cost (₹/kWh)',
    subsidy_amount  DECIMAL(12,2)   NOT NULL DEFAULT 0.00    COMMENT 'Government subsidy for residential solar (₹)',
    net_metering    BOOLEAN         NOT NULL DEFAULT TRUE     COMMENT 'Whether the state supports net metering',
    effective_date  DATE            NOT NULL                  COMMENT 'Date from which this tariff is effective',
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_state (state_name)
) ENGINE=InnoDB COMMENT='State-level electricity tariffs and solar subsidies';


-- ---------------------------------------------------------------------------
-- Table 2: user_queries
-- ---------------------------------------------------------------------------
-- Immutable audit log of every prediction request made through the Gradio
-- frontend or the Java API gateway.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_queries (
    id                  BIGINT          AUTO_INCREMENT PRIMARY KEY,
    session_id          CHAR(36)        NOT NULL                     COMMENT 'UUID v4 identifying the user session',
    latitude            DECIMAL(9,6)    NOT NULL                     COMMENT 'Query latitude',
    longitude           DECIMAL(9,6)    NOT NULL                     COMMENT 'Query longitude',
    roof_area_sqm       DECIMAL(8,2)    NOT NULL                     COMMENT 'Roof area in square metres',
    monthly_bill        DECIMAL(10,2)   NOT NULL                     COMMENT 'Current monthly electricity bill (₹)',
    predicted_roi_pct   DECIMAL(6,2)    DEFAULT NULL                 COMMENT 'Model-predicted ROI percentage',
    predicted_payback_yr DECIMAL(4,1)   DEFAULT NULL                 COMMENT 'Estimated payback period in years',
    is_viable           BOOLEAN         DEFAULT NULL                 COMMENT 'Binary classification: viable for solar?',
    model_version       VARCHAR(50)     DEFAULT NULL                 COMMENT 'ML model version used for prediction',
    state_tariff_id     INT             DEFAULT NULL                 COMMENT 'FK to the tariff row used in calculation',
    created_at          TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_session   (session_id),
    INDEX idx_coords    (latitude, longitude),
    INDEX idx_created   (created_at),

    CONSTRAINT fk_state_tariff
        FOREIGN KEY (state_tariff_id) REFERENCES state_tariffs(id)
        ON DELETE SET NULL
        ON UPDATE CASCADE
) ENGINE=InnoDB COMMENT='Audit log of user predictions';


-- ---------------------------------------------------------------------------
-- Seed data: sample tariffs for a few Indian states
-- ---------------------------------------------------------------------------
INSERT INTO state_tariffs (state_name, electricity_rate, subsidy_amount, net_metering, effective_date)
VALUES
    ('Karnataka',       6.95,  94822.00, TRUE,  '2024-04-01'),
    ('Maharashtra',     9.30,  78000.00, TRUE,  '2024-04-01'),
    ('Tamil Nadu',      4.50,  94822.00, TRUE,  '2024-04-01'),
    ('Delhi',           8.00,  94822.00, TRUE,  '2024-04-01'),
    ('Rajasthan',       7.55,  78000.00, TRUE,  '2024-04-01'),
    ('Gujarat',         5.50,  78000.00, TRUE,  '2024-04-01'),
    ('Uttar Pradesh',   7.00,  78000.00, TRUE,  '2024-04-01'),
    ('West Bengal',     8.20,  78000.00, FALSE, '2024-04-01'),
    ('Kerala',          6.80,  78000.00, TRUE,  '2024-04-01'),
    ('Assam',           6.15,  78000.00, FALSE, '2024-04-01')
ON DUPLICATE KEY UPDATE
    electricity_rate = VALUES(electricity_rate),
    subsidy_amount   = VALUES(subsidy_amount);
