"""
HRV and Baevsky formula constants.

Units are documented on each constant. Sources:
  Kubios HRV User Guide 3.x (Baevsky histogram bin width)
  Task Force of the European Society of Cardiology (1996), HRV standards
"""

# Valid inter-beat interval range for adults at rest (milliseconds).
MIN_INTERBEAT_INTERVAL_MS = 300
MAX_INTERBEAT_INTERVAL_MS = 2000

# Reject IBI if it deviates more than this fraction from the window median.
OUTLIER_MEDIAN_DEVIATION_RATIO = 0.20

# Max allowed gap between ibi_ts and next IBI value when pairing for RMSSD (ms).
MAX_TIMESTAMP_IBI_MISMATCH_MS = 250

# Minimum beats and covered time before HRV metrics are considered valid.
MIN_BEATS_FOR_HRV = 8
MIN_COVERED_SECONDS_FOR_HRV = 6.0

# Server-side IBI lookback when trimming epoch payloads (ms). Watch may retain 120s.
IBI_LOOKBACK_TRIM_MS = 60_000

# Rolling buffer retention on server (ms).
IBI_BUFFER_RETENTION_MS = 120_000

# Baevsky histogram bin width (ms). Kubios default is 50 ms.
BAEVSKY_HISTOGRAM_BIN_MS = 50

# Population reference for sqrt(Baevsky) stress index at rest (Kubios guide).
SQRT_BAEVSKY_NORMAL_LOW = 7.0
SQRT_BAEVSKY_NORMAL_HIGH = 12.0

# Parallel analysis window lengths (seconds).
ANALYSIS_WINDOW_SECONDS = (15, 30, 60, 90)

# Primary window used for UI decisions (seconds).
PRIMARY_DECISION_WINDOW_SECONDS = 30

# Epoch publish interval on watch (seconds).
EPOCH_PUBLISH_INTERVAL_SECONDS = 30
