from __future__ import annotations

import openml

openml.config.server = "https://www.openml.org/api/v1/xml"
setups = openml.setups.list_setups(flow=5873)

print(len(setups))
print(setups)
