"""Bundle the frozen public archive and reader UI into one offline HTML file.

No live services or experiment files are accessed. Rebuild from this directory:
    python build_lineage.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Produce the portable reader view from its frozen public inputs."""
    template = (ROOT / "lineage-template.html").read_text()
    data = json.loads((ROOT / "lineage-data.json").read_text())
    replacements = {
        "/* INLINE_STYLE */": (ROOT / "lineage-explorer.css").read_text(),
        "/* INLINE_SCRIPT */": (ROOT / "lineage-explorer.js").read_text(),
        '"__INLINE_DATA__"': json.dumps(data, ensure_ascii=False, separators=(",", ":")),
    }
    for marker, value in replacements.items():
        # Data and source must not terminate their containing HTML element.
        if marker != "/* INLINE_STYLE */":
            value = value.replace("</", "<\\/")
        if template.count(marker) != 1:
            raise ValueError(f"Expected one template marker: {marker}")
        template = template.replace(marker, value)
    (ROOT / "lineage-explorer.html").write_text(template)
    sys.stdout.write(f"Bundled {len(data['nodes'])} nodes and {len(data['edges'])} edges.\n")


if __name__ == "__main__":
    main()
