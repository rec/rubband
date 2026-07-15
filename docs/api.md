# API Reference

This page is generated from the public Python API in `rubband`.

## Configuration

::: rubband.Options
    options:
      heading_level: 3
      members:
        - option_flags

::: rubband.RubberBandMetadata
    options:
      heading_level: 3

## Stateful Processing

::: rubband.Stretcher
    options:
      heading_level: 3
      members:
        - study
        - process
        - reset
        - set_time_ratio
        - set_pitch_scale
        - set_formant_scale
        - set_transients_option
        - set_detector_option
        - set_phase_option
        - set_formant_option
        - set_pitch_option
        - get_time_ratio
        - get_pitch_scale
        - get_formant_scale
        - get_preferred_start_pad
        - get_start_delay
        - get_latency
        - get_channel_count
        - set_expected_input_duration
        - set_max_process_size
        - get_process_size_limit
        - get_samples_required
        - available
        - retrieve

## Convenience Functions

::: rubband.stretch
    options:
      heading_level: 3

::: rubband.metadata
    options:
      heading_level: 3

## Option Enums

The enum values below map directly to Rubber Band option groups.

### `PresetOption`

| Value | Meaning |
| --- | --- |
| `default` | Use explicit option values. |
| `percussive` | Apply Rubber Band's percussive preset. |

### `ProcessOption`

| Value | Meaning |
| --- | --- |
| `offline` | Offline processing. Time and pitch ratios are fixed after `study()` or `process()` starts. |
| `real_time` | Real-time processing. Time and pitch ratios may change while processing. |

### `StretchOption`

| Value | Meaning |
| --- | --- |
| `elastic` | Standard Rubber Band elastic stretching. |
| `precise` | Prefer precise stretching behavior. |

### `TransientsOption`

| Value | Meaning |
| --- | --- |
| `crisp` | Preserve transients more sharply. |
| `mixed` | Balance transient preservation and smoothing. |
| `smooth` | Smooth transients more heavily. |

### `DetectorOption`

| Value | Meaning |
| --- | --- |
| `compound` | Use Rubber Band's compound transient detector. |
| `percussive` | Tune detection for percussive material. |
| `soft` | Tune detection for softer material. |

### `PhaseOption`

| Value | Meaning |
| --- | --- |
| `laminar` | Keep phase behavior more coherent. |
| `independent` | Allow channels or components to phase independently. |

### `ThreadingOption`

| Value | Meaning |
| --- | --- |
| `auto` | Let Rubber Band choose threading behavior. |
| `never` | Disable Rubber Band's internal threading. |
| `always` | Enable Rubber Band's internal threading. |

### `WindowOption`

| Value | Meaning |
| --- | --- |
| `standard` | Use the standard analysis window. |
| `short` | Use a shorter analysis window. |
| `long` | Use a longer analysis window. |

### `SmoothingOption`

| Value | Meaning |
| --- | --- |
| `off` | Disable Rubber Band's smoothing option. |
| `on` | Enable Rubber Band's smoothing option. |

### `FormantOption`

| Value | Meaning |
| --- | --- |
| `shifted` | Shift formants with pitch. |
| `preserved` | Preserve formants while shifting pitch. |

### `PitchOption`

| Value | Meaning |
| --- | --- |
| `high_speed` | Prefer faster pitch processing. |
| `high_quality` | Prefer higher quality pitch processing. |
| `high_consistency` | Prefer more consistent pitch processing. |

### `ChannelsOption`

| Value | Meaning |
| --- | --- |
| `apart` | Process channels separately. |
| `together` | Process channels together. |

### `EngineOption`

| Value | Meaning |
| --- | --- |
| `faster` | Use Rubber Band's faster engine. |
| `finer` | Use Rubber Band's finer engine. |
