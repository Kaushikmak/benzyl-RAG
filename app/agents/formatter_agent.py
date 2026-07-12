"""FormatterAgent responsible for formatting output report into target format."""


class FormatterAgent:
    """Formats verified Markdown synthesis into target output format."""

    def format(self, report: str, target_format: str = "Markdown") -> str:
        """Format report string into specified target format."""
        fmt = target_format.upper()
        if fmt in ("MARKDOWN", "MD"):
            return report.strip()
        elif fmt == "HTML":
            return f"<article>\n{report.strip()}\n</article>"
        elif fmt == "JSON":
            import json
            return json.dumps({"report": report.strip()})
        return report.strip()
