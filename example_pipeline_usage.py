"""
Example script demonstrating how to use the pipeline storage classes.
This shows how to interact with both SQLite and Supabase storage without running full training.
"""

import os
import tempfile
from datetime import datetime

print("="*70)
print("Pipeline Storage Usage Examples")
print("="*70)

# Example 1: SQLite Storage
print("\n" + "="*70)
print("Example 1: SQLite Storage")
print("="*70)

from updated_pipeline import SQLiteStorage

# Create a temporary database for demo
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = os.path.join(tmpdir, 'demo.db')
    print(f"\nCreating SQLite database: {db_path}")
    
    # Initialize storage
    storage = SQLiteStorage(db_path)
    print("✓ Storage initialized")
    
    # Create an experiment
    config = {
        'backbone': 'dinov2-vit-b',
        'batch_size': 32,
        'lr': 0.0001,
        'epochs': 100,
        'training_mode': 'multi_stage'
    }
    exp_id = storage.create_experiment('demo_experiment', config)
    print(f"✓ Created experiment with ID: {exp_id}")
    
    # Log some training metrics
    print("\nLogging training metrics...")
    for epoch in range(1, 6):
        # Simulate improving metrics
        loss = 1.0 / epoch
        accuracy = 0.5 + (epoch * 0.05)
        
        storage.log_metric(exp_id, epoch, 1, 'loss', loss, 'train')
        storage.log_metric(exp_id, epoch, 1, 'accuracy', accuracy, 'train')
        print(f"  Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")
    
    # Log evaluation results
    print("\nLogging evaluation results...")
    eval_results = {
        'accuracy': 0.85,
        'top3_accuracy': 0.92,
        'top5_accuracy': 0.95,
        'f1_macro': 0.83,
        'f1_weighted': 0.84,
        'paired_accuracy': 0.87,
        'unpaired_accuracy': 0.82
    }
    storage.log_evaluation(exp_id, 5, eval_results)
    print(f"✓ Logged evaluation: accuracy={eval_results['accuracy']:.4f}")
    
    # Save checkpoint info
    checkpoint_path = '/path/to/model_epoch5.pth'
    storage.save_checkpoint_info(exp_id, 5, checkpoint_path, 
                                is_best=True, metric_value=0.85)
    print(f"✓ Saved checkpoint info: {checkpoint_path}")
    
    # Update experiment status
    storage.update_experiment_status(exp_id, 'completed')
    print("✓ Updated experiment status to 'completed'")
    
    # Query results
    print("\nQuerying results...")
    metrics = storage.get_experiment_metrics(exp_id)
    print(f"✓ Retrieved {len(metrics)} metric records")
    
    best_checkpoint = storage.get_best_checkpoint(exp_id)
    if best_checkpoint:
        print(f"✓ Best checkpoint: epoch {best_checkpoint['epoch']}, "
              f"accuracy {best_checkpoint['metric_value']:.4f}")
    
    storage.close()
    print("\n✓ SQLite example completed successfully!")

# Example 2: Supabase Storage (demonstration without actual connection)
print("\n" + "="*70)
print("Example 2: Supabase Storage (Pseudo-code)")
print("="*70)

print("""
# To use Supabase storage, you would:

from updated_pipeline_supabase import SupabaseStorage

# Initialize with your credentials
storage = SupabaseStorage(
    url='https://your-project.supabase.co',
    key='your-api-key',
    table_prefix='ml_pipeline'
)

# The API is identical to SQLite
exp_id = storage.create_experiment('my_experiment', config)

# Log metrics (same interface)
storage.log_metric(exp_id, epoch=1, stage=1, 
                  metric_name='loss', metric_value=0.5, metric_type='train')

# Log evaluations (same interface)
storage.log_evaluation(exp_id, epoch=5, results={
    'accuracy': 0.85,
    'f1_macro': 0.83,
    # ... more metrics
})

# Save checkpoint info (same interface)
storage.save_checkpoint_info(exp_id, epoch=5, checkpoint_path='/path/to/model.pth',
                            is_best=True, metric_value=0.85)

# Update status (same interface)
storage.update_experiment_status(exp_id, 'completed')

# Query results (same interface)
metrics = storage.get_experiment_metrics(exp_id)
best = storage.get_best_checkpoint(exp_id)
""")

print("Note: The Supabase API is identical to SQLite, just with cloud storage!")
print("✓ See PIPELINE_USAGE.md for complete Supabase setup instructions")

# Example 3: Comparison
print("\n" + "="*70)
print("Example 3: When to Use Each Storage Backend")
print("="*70)

comparison = """
SQLite Storage:
  ✓ Local development and testing
  ✓ Single-machine training
  ✓ Quick experimentation
  ✓ No external dependencies
  ✓ Simple backup (just copy .db file)
  
Supabase Storage:
  ✓ Production deployments
  ✓ Multi-machine experiments
  ✓ Team collaboration
  ✓ Real-time monitoring
  ✓ Web dashboard access
  ✓ Automatic backups
"""

print(comparison)

print("\n" + "="*70)
print("Examples completed!")
print("="*70)
print("\nNext steps:")
print("  1. Try: python example_pipeline_usage.py")
print("  2. Read: PIPELINE_USAGE.md for detailed instructions")
print("  3. Run: python updated_pipeline.py --help")
print("  4. Run: python updated_pipeline_supabase.py --help")
print("="*70)
