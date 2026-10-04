"""Inject outputs/results.json into dashboard_template.html -> dashboard.html (self-contained)."""
import json
data = open("outputs/results.json").read()
html = open("dashboard_template.html").read().replace("__DATA__", data)
open("dashboard.html", "w").write(html)
print("dashboard.html", round(len(html) / 1024), "KB")
