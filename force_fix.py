import os

path = 'templates/leaderboard.html'
with open(path, 'r') as f:
    content = f.read()

# The broken string pattern
broken = 'const leaderboardId = {{ leaderboard.id }\n    };'
# The fixed string pattern
fixed = 'const leaderboardId = {{ leaderboard.id }};'

if broken in content:
    print(f"Broken pattern found in {path}. Fixing...")
    new_content = content.replace(broken, fixed)
    with open(path, 'w') as f:
        f.write(new_content)
    print("File updated.")
else:
    print(f"Broken pattern NOT found in {path}.")
    # Check for the line specifically
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if 'const leaderboardId = {{ leaderboard.id }' in line and '};' not in line:
             print(f"Found broken line at {i+1}: {line}")
             lines[i] = '        const leaderboardId = {{ leaderboard.id }};'
             # Remove next line if it is just };
             if i+1 < len(lines) and lines[i+1].strip() == '};':
                 lines.pop(i+1)
             
             with open(path, 'w') as f:
                 f.write('\n'.join(lines))
             print("Fixed line based.")
             break

