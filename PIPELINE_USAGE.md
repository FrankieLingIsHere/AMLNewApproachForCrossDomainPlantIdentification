# Pipeline Usage Guide

This document explains how to use the updated training pipelines with database storage.

## Overview

Two pipeline scripts are available for training and evaluation:

1. **`updated_pipeline.py`** - Uses SQLite for local storage of metrics and results
2. **`updated_pipeline_supabase.py`** - Standalone version using Supabase cloud storage

Both pipelines provide:
- Complete training workflow
- Automatic metric tracking
- Evaluation at regular intervals
- Checkpoint saving
- Database storage of all results

## SQLite Pipeline (`updated_pipeline.py`)

### Features
- Local SQLite database storage
- No external dependencies or credentials needed
- Perfect for local development and testing
- All data stored in a single `.db` file

### Installation
```bash
# No additional dependencies needed beyond base requirements
pip install -r requirements.txt
```

### Usage

Basic usage:
```bash
python updated_pipeline.py --epochs 50
```

With custom settings:
```bash
python updated_pipeline.py \
    --backbone dinov2-vit-b \
    --batch_size 32 \
    --epochs 100 \
    --mixed_precision \
    --db_path my_results.db \
    --exp_name my_experiment
```

### Database Schema

The SQLite database contains four tables:

1. **experiments**: Experiment metadata and configuration
2. **metrics**: Training metrics per epoch
3. **evaluation_results**: Evaluation metrics at intervals
4. **checkpoints**: Information about saved model checkpoints

### Viewing Results

You can query the SQLite database using any SQLite client or Python:

```python
import sqlite3

conn = sqlite3.connect('pipeline_results.db')
cursor = conn.cursor()

# Get all experiments
cursor.execute("SELECT * FROM experiments")
experiments = cursor.fetchall()

# Get metrics for a specific experiment
cursor.execute("SELECT * FROM metrics WHERE exp_id = 1 ORDER BY epoch")
metrics = cursor.fetchall()

conn.close()
```

## Supabase Pipeline (`updated_pipeline_supabase.py`)

### Features
- Cloud-based storage with Supabase
- Real-time collaboration and monitoring
- Web dashboard access to all experiments
- Scalable and persistent storage
- Standalone - completely independent implementation

### Prerequisites

1. Install Supabase client:
```bash
pip install supabase
```

2. Create a Supabase project at [supabase.com](https://supabase.com)

3. Create the required tables in your Supabase project:

```sql
-- Experiments table
CREATE TABLE ml_pipeline_experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exp_name TEXT UNIQUE NOT NULL,
    backbone TEXT,
    batch_size INTEGER,
    learning_rate REAL,
    epochs INTEGER,
    training_mode TEXT,
    config JSONB,
    status TEXT DEFAULT 'running',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Metrics table
CREATE TABLE ml_pipeline_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exp_id UUID REFERENCES ml_pipeline_experiments(id),
    epoch INTEGER,
    stage INTEGER,
    metric_name TEXT,
    metric_value REAL,
    metric_type TEXT,
    recorded_at TIMESTAMP DEFAULT NOW()
);

-- Evaluations table
CREATE TABLE ml_pipeline_evaluations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exp_id UUID REFERENCES ml_pipeline_experiments(id),
    epoch INTEGER,
    accuracy REAL,
    top3_accuracy REAL,
    top5_accuracy REAL,
    f1_macro REAL,
    f1_weighted REAL,
    paired_accuracy REAL,
    unpaired_accuracy REAL,
    results_json JSONB,
    evaluated_at TIMESTAMP DEFAULT NOW()
);

-- Checkpoints table
CREATE TABLE ml_pipeline_checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exp_id UUID REFERENCES ml_pipeline_experiments(id),
    epoch INTEGER,
    checkpoint_path TEXT,
    is_best BOOLEAN DEFAULT FALSE,
    metric_value REAL,
    saved_at TIMESTAMP DEFAULT NOW()
);

-- Add indexes for better query performance
CREATE INDEX idx_metrics_exp_id ON ml_pipeline_metrics(exp_id);
CREATE INDEX idx_evaluations_exp_id ON ml_pipeline_evaluations(exp_id);
CREATE INDEX idx_checkpoints_exp_id ON ml_pipeline_checkpoints(exp_id);
```

### Usage

#### Method 1: Environment Variables (Recommended)
```bash
# Set credentials
export SUPABASE_URL=https://your-project.supabase.co
export SUPABASE_KEY=your-api-key

# Run pipeline
python updated_pipeline_supabase.py --epochs 50
```

#### Method 2: Command Line Arguments
```bash
python updated_pipeline_supabase.py \
    --supabase_url https://your-project.supabase.co \
    --supabase_key your-api-key \
    --epochs 50
```

#### Full Configuration Example
```bash
python updated_pipeline_supabase.py \
    --supabase_url https://your-project.supabase.co \
    --supabase_key your-api-key \
    --backbone dinov2-vit-b \
    --batch_size 32 \
    --epochs 100 \
    --mixed_precision \
    --exp_name production_run_v1 \
    --table_prefix ml_pipeline
```

### Viewing Results

Access your results through:

1. **Supabase Dashboard**: View tables directly in your Supabase project
2. **SQL Editor**: Run custom queries
3. **Python Client**:

```python
from supabase import create_client

url = "https://your-project.supabase.co"
key = "your-api-key"
supabase = create_client(url, key)

# Get all experiments
experiments = supabase.table('ml_pipeline_experiments').select("*").execute()

# Get metrics for a specific experiment
metrics = supabase.table('ml_pipeline_metrics').select("*").eq('exp_id', 'exp-uuid').execute()
```

## Command Line Arguments

### Common Arguments (Both Pipelines)

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `Herbarium_Field` | Path to dataset |
| `--batch_size` | `32` | Training batch size |
| `--num_workers` | `4` | Data loading workers |
| `--image_size` | `224` | Input image size |
| `--backbone` | `dinov2-vit-b` | Model backbone |
| `--pretrained` | `True` | Use pretrained weights |
| `--dropout` | `0.1` | Dropout rate |
| `--epochs` | `50` | Number of epochs |
| `--lr` | `1e-4` | Learning rate |
| `--training_mode` | `single_stage` | Training mode |
| `--mixed_precision` | `False` | Use mixed precision |
| `--exp_name` | Auto-generated | Experiment name |
| `--device` | `cuda` | Device to use |
| `--seed` | `42` | Random seed |

### SQLite-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--db_path` | `pipeline_results.db` | Path to SQLite database |

### Supabase-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--supabase_url` | From env | Supabase project URL |
| `--supabase_key` | From env | Supabase API key |
| `--table_prefix` | `ml_pipeline` | Table name prefix |

## Comparison: SQLite vs Supabase

| Feature | SQLite | Supabase |
|---------|--------|----------|
| Setup Complexity | Simple | Moderate |
| External Dependencies | None | Supabase account |
| Data Persistence | Local file | Cloud |
| Collaboration | Limited | Real-time |
| Scalability | Single machine | Highly scalable |
| Web Dashboard | No | Yes |
| Cost | Free | Free tier + paid |
| Best For | Local dev, testing | Production, teams |

## Tips and Best Practices

### For SQLite Pipeline

1. **Backup your database**: Copy the `.db` file regularly
2. **Use descriptive experiment names**: Makes querying easier
3. **Single machine**: SQLite is perfect for local development
4. **Version control**: Add `*.db` to `.gitignore`

### For Supabase Pipeline

1. **Use environment variables**: Never commit API keys
2. **Set up Row Level Security (RLS)**: Protect your data
3. **Monitor usage**: Check your Supabase project limits
4. **Use table prefixes**: Helps organize multiple projects
5. **Enable real-time**: Monitor experiments as they run

## Troubleshooting

### SQLite Issues

**Database locked error:**
```bash
# Close all connections to the database
# Only one process should write at a time
```

**Database file not found:**
```bash
# The database is created automatically on first run
# Check file permissions in the directory
```

### Supabase Issues

**Connection failed:**
- Check your URL and API key
- Ensure tables are created
- Check network connectivity

**Insert failed:**
- Verify table schema matches expected structure
- Check for unique constraint violations
- Review Supabase logs in dashboard

**Missing supabase module:**
```bash
pip install supabase
```

## Examples

### Quick Start with SQLite
```bash
python updated_pipeline.py \
    --epochs 10 \
    --batch_size 16 \
    --exp_name quick_test
```

### Production Run with Supabase
```bash
export SUPABASE_URL=https://xxx.supabase.co
export SUPABASE_KEY=your-key

python updated_pipeline_supabase.py \
    --backbone dinov2-vit-l \
    --batch_size 64 \
    --epochs 100 \
    --mixed_precision \
    --exp_name production_v1
```

### Compare Multiple Experiments (SQLite)
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('pipeline_results.db')

# Get best accuracy for each experiment
query = """
SELECT e.exp_name, MAX(ev.accuracy) as best_accuracy
FROM experiments e
JOIN evaluation_results ev ON e.id = ev.exp_id
GROUP BY e.exp_name
ORDER BY best_accuracy DESC
"""

df = pd.read_sql_query(query, conn)
print(df)
```

## Additional Resources

- [Supabase Documentation](https://supabase.com/docs)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- Main project README: [README.md](README.md)
- Training guide: [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)

## Support

For issues or questions:
1. Check this documentation
2. Review error messages carefully
3. Open a GitHub issue with details
