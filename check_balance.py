import re

with open('templates/comparison.html', 'r') as f:
    lines = f.readlines()

in_script = False
script_content = ""
script_lines = []

for i, line in enumerate(lines):
    if '<script>' in line:
        in_script = True
        continue
    if '</script>' in line:
        in_script = False
        continue
    if in_script:
        script_content += line
        script_lines.append((i + 1, line))

# Check parens, braces, brackets balance
stack = []
pairs = {')': '(', '}': '{', ']': '['}

for line_num, line in script_lines:
    # Ignore strings
    cleaned_line = re.sub(r"'.*?'", "''", line)
    cleaned_line = re.sub(r'".*?"', '""', cleaned_line)
    
    for char in cleaned_line:
        if char in '({[':
            stack.append((char, line_num))
        elif char in ')}]':
            if not stack:
                print(f"Error: Unexpected {char} at line {line_num}")
                continue
            top_char, top_line = stack.pop()
            if top_char != pairs[char]:
                print(f"Error: Mismatched {char} at line {line_num} (opens with {top_char} at line {top_line})")

for char, line_num in stack:
    print(f"Error: Unclosed {char} starting at line {line_num}")
