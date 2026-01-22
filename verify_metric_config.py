from app import app, db, Leaderboard
import json
import os

# Set up context
with app.app_context():
    print("Verifying Leaderboard model...")
    # Check if column exists in DB
    try:
        # Create a dummy leaderboard
        lb = Leaderboard(name="Test Config LB", dataset_id=1, summary_metrics="l1")
        db.session.add(lb)
        db.session.commit()
        
        print(f"Created leaderboard {lb.id}")
        
        # Test default
        print(f"Default metric_directions: '{lb.metric_directions}'")
        assert lb.metric_directions == '{}' or lb.metric_directions is None 
        
        # Test updating
        directions = {'l1': 'lower_is_better', 'peak': 'higher_is_better'}
        lb.metric_directions = json.dumps(directions)
        db.session.commit()
        
        # Test retrieval
        lb_fetched = Leaderboard.query.get(lb.id)
        print(f"Fetched metric_directions: {lb_fetched.metric_directions}")
        loaded = json.loads(lb_fetched.metric_directions)
        assert loaded['l1'] == 'lower_is_better'
        assert loaded['peak'] == 'higher_is_better'
        
        # Cleanup
        db.session.delete(lb)
        db.session.commit()
        print("Verification successful!")
        
    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()
