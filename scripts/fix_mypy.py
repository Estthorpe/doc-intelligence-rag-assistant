# scripts/fix_mypy.py
"""Apply remaining mypy fixes."""

import re

# Fix 1: settings.py
with open("src/config/settings.py") as f:
    content = f.read()
if "# type: ignore[call-arg]" not in content:
    content = content.replace(
        "settings = Settings()",
        "settings = Settings()  # type: ignore[call-arg]",
    )
    with open("src/config/settings.py", "w") as f:
        f.write(content)
    print("Fixed settings.py")
else:
    print("settings.py already fixed")

# Fix 2: contracts.py - dict -> dict[str, Any]
with open("src/ingestion/contracts.py") as f:
    content = f.read()
if "from typing import" not in content:
    content = "from typing import Any\n" + content
elif "Any" not in content:
    content = re.sub(
        r"from typing import ([^\n]+)",
        r"from typing import \1, Any",
        content,
    )
content = content.replace("metadata: dict =", "metadata: dict[str, Any] =")
with open("src/ingestion/contracts.py", "w") as f:
    f.write(content)
print("Fixed contracts.py")

# Fix 3: indexer.py - explicit int return
with open("src/ingestion/indexer.py") as f:
    content = f.read()
if "count: int =" not in content:
    content = content.replace(
        "return result[0] if result else 0",
        "count: int = result[0] if result else 0\n            return count",
    )
    with open("src/ingestion/indexer.py", "w") as f:
        f.write(content)
    print("Fixed indexer.py")
else:
    print("indexer.py already fixed")

print("\nAll fixes applied.")
