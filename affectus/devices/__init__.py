"""
Device families. Each family (wrist / chest / head) is a self-contained module
that declares: the threshold profile it uses, the adapter that turns its raw wire
payload into a SensorFrame, and the per-sensor feature-modules it can run.

Importing this package imports every family, which registers their profiles
(shared.profiles) and makes their modules available to devices.registry. Only the
wrist family is implemented today; chest and head are templates.
"""

from affectus.devices import wrist  # noqa: F401  (registers wrist_ppg profile)

# Templates — import so their (empty) registration runs; no behaviour yet.
from affectus.devices import chest  # noqa: F401
from affectus.devices import head   # noqa: F401
