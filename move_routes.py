import os

path = 'app.py'
with open(path, 'r') as f:
    content = f.read()

# Marker for the start of the misplaced block
start_marker = "@app.route('/api/leaderboard/<int:leaderboard_id>/recalculate_async', methods=['POST'])"
# Marker for the insertion point
insert_marker = "@app.template_filter('from_json')"

if start_marker in content and insert_marker in content:
    # Split content
    parts = content.split(start_marker)
    if len(parts) == 2:
        pre_misplaced = parts[0]
        misplaced_block = start_marker + parts[1]
        
        # Now insert before insert_marker in pre_misplaced
        if insert_marker in pre_misplaced:
            final_parts = pre_misplaced.split(insert_marker)
            new_content = final_parts[0] + misplaced_block + "\n\n" + insert_marker + final_parts[1]
            
            with open(path, 'w') as f:
                f.write(new_content)
            print("Successfully moved routes.")
        else:
            print("Insertion point not found in the upper part of the file.")
    else:
        print("Multiple or no occurrences of start marker.")
else:
    print("Markers not found.")
