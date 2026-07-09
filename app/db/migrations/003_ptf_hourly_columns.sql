DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ptf_hourly' AND column_name = 'delivery_time'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ptf_hourly' AND column_name = 'timestamp'
    ) THEN
        ALTER TABLE ptf_hourly RENAME COLUMN delivery_time TO "timestamp";
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ptf_hourly' AND column_name = 'price_try_mwh'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ptf_hourly' AND column_name = 'ptf_tl'
    ) THEN
        ALTER TABLE ptf_hourly RENAME COLUMN price_try_mwh TO ptf_tl;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ptf_hourly' AND column_name = 'price_usd_mwh'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ptf_hourly' AND column_name = 'ptf_usd'
    ) THEN
        ALTER TABLE ptf_hourly RENAME COLUMN price_usd_mwh TO ptf_usd;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ptf_hourly' AND column_name = 'price_eur_mwh'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ptf_hourly' AND column_name = 'ptf_eur'
    ) THEN
        ALTER TABLE ptf_hourly RENAME COLUMN price_eur_mwh TO ptf_eur;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ptf_hourly' AND column_name = 'ingested_at'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ptf_hourly' AND column_name = 'created_at'
    ) THEN
        ALTER TABLE ptf_hourly RENAME COLUMN ingested_at TO created_at;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ptf_hourly' AND column_name = 'source_updated_at'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ptf_hourly' AND column_name = 'updated_at'
    ) THEN
        ALTER TABLE ptf_hourly RENAME COLUMN source_updated_at TO updated_at;
    END IF;
END
$$;

ALTER TABLE ptf_hourly
    ADD COLUMN IF NOT EXISTS source TEXT,
    ADD COLUMN IF NOT EXISTS raw_record JSONB,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

UPDATE ptf_hourly
SET source = COALESCE(source, 'epias'),
    raw_record = COALESCE(raw_record, '{}'::JSONB),
    created_at = COALESCE(created_at, NOW()),
    updated_at = COALESCE(updated_at, created_at, NOW());

ALTER TABLE ptf_hourly
    ALTER COLUMN ptf_tl TYPE NUMERIC USING ptf_tl::NUMERIC,
    ALTER COLUMN ptf_usd TYPE NUMERIC USING ptf_usd::NUMERIC,
    ALTER COLUMN ptf_eur TYPE NUMERIC USING ptf_eur::NUMERIC,
    ALTER COLUMN source SET DEFAULT 'epias',
    ALTER COLUMN source SET NOT NULL,
    ALTER COLUMN raw_record SET DEFAULT '{}'::JSONB,
    ALTER COLUMN raw_record SET NOT NULL,
    ALTER COLUMN created_at SET DEFAULT NOW(),
    ALTER COLUMN created_at SET NOT NULL,
    ALTER COLUMN updated_at SET DEFAULT NOW(),
    ALTER COLUMN updated_at SET NOT NULL;
