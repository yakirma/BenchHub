"""
Migration script to enable histogram entropy for existing leaderboards.
"""
from app import db, app, Leaderboard

with app.app_context():
    # Find leaderboards with empty selected_metrics
    leaderboards_without_metrics = Leaderboard.query.filter(
        (Leaderboard.selected_metrics == '') | (Leaderboard.selected_metrics is None)
    ).all()
    
    print(f"Found {len(leaderboards_without_metrics)} leaderboards without metrics enabled")
    
    if leaderboards_without_metrics:
        for lb in leaderboards_without_metrics:
            lb.selected_metrics = 'hist_entropy'
            print(f"  - Enabled hist_entropy for leaderboard: {lb.name}")
        
        db.session.commit()
        print(f"✅ Updated {len(leaderboards_without_metrics)} leaderboards")
    else:
        print("✅ No leaderboards need updating")
