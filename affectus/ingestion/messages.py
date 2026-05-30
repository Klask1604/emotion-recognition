"""Raw input messages from the watch (acquisition/batch v2)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InterbeatIntervalEntry:
    interval_ms: int
    timestamp_ms: int | None = None


@dataclass
class IbiBatchMessage:
    """One second of IBI data from the watch."""

    timestamp_ms: int
    intervals_ms: list[int] = field(default_factory=list)
    timestamps_ms: list[int] = field(default_factory=list)


@dataclass
class PpgBatchMessage:
    """One second of PPG samples from the watch."""

    timestamp_ms: int
    green: list[int] = field(default_factory=list)
    infrared: list[int] = field(default_factory=list)
    sample_timestamps_ms: list[int] = field(default_factory=list)


@dataclass
class AcquisitionBatchMessage:
    """Atomic 1 Hz acquisition frame (schema v2) from the watch."""

    timestamp_publish_ms: int
    timestamp_anchor_ms: int
    sequence: int
    heart_rate_bpm: float = 0.0
    display_on: bool = True
    skin_temperature_c: float = 0.0
    ambient_temperature_c: float = 0.0
    skin_temperature_ts_ms: int = 0
    acceleration_rms: float = 0.0
    acceleration_p90: float = 0.0
    acceleration_std: float = 0.0
    gyroscope_rms: float = 0.0
    gyroscope_p90: float = 0.0
    gyroscope_std: float = 0.0
    # Cardiac-band (0.5-4 Hz) acceleration energy from the watch: the part of
    # wrist motion that overlaps and corrupts the PPG pulse. Drives the
    # signal-quality gate. Falls back to acceleration_rms when absent.
    acc_band_cardiac: float = 0.0
    motion_window_ms: int = 1000
    ibi_intervals_ms: list[int] = field(default_factory=list)
    ibi_timestamps_ms: list[int] = field(default_factory=list)
    ibi_timestamp_source: str = "reconstructed"
    # Raw PPG (research/legacy only; absent unless the watch toggle is on).
    ppg_green: list[int] = field(default_factory=list)
    ppg_infrared: list[int] = field(default_factory=list)
    ppg_timestamps_ms: list[int] = field(default_factory=list)

    def motion_energy(self) -> float:
        """Motion-artifact proxy for the signal-quality gate."""
        return self.acc_band_cardiac if self.acc_band_cardiac > 0 else self.acceleration_rms

    def to_sensor_batch(self) -> SensorBatchMessage:
        return SensorBatchMessage(
            timestamp_ms=self.timestamp_anchor_ms,
            heart_rate_bpm=self.heart_rate_bpm,
            acceleration_rms=self.acceleration_rms,
            acceleration_p90=self.acceleration_p90,
            acceleration_std=self.acceleration_std,
            gyroscope_rms=self.gyroscope_rms,
            gyroscope_p90=self.gyroscope_p90,
            gyroscope_std=self.gyroscope_std,
            skin_temperature_c=self.skin_temperature_c,
            ambient_temperature_c=self.ambient_temperature_c,
            display_on=self.display_on,
        )

    def to_ibi_batch(self) -> IbiBatchMessage:
        return IbiBatchMessage(
            timestamp_ms=self.timestamp_anchor_ms,
            intervals_ms=self.ibi_intervals_ms,
            timestamps_ms=self.ibi_timestamps_ms,
        )


@dataclass
class SensorBatchMessage:
    """Aggregated sensor stats from the watch (1 Hz)."""

    timestamp_ms: int
    heart_rate_bpm: float = 0.0
    acceleration_rms: float = 0.0
    acceleration_p90: float = 0.0
    acceleration_std: float = 0.0
    gyroscope_rms: float = 0.0
    gyroscope_p90: float = 0.0
    gyroscope_std: float = 0.0
    skin_temperature_c: float = 0.0
    ambient_temperature_c: float = 0.0
    display_on: bool = True
