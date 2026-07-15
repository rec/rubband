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

::: rubband.ProcessOption
    options:
      heading_level: 3

::: rubband.StretchOption
    options:
      heading_level: 3

::: rubband.TransientsOption
    options:
      heading_level: 3

::: rubband.DetectorOption
    options:
      heading_level: 3

::: rubband.PhaseOption
    options:
      heading_level: 3

::: rubband.ThreadingOption
    options:
      heading_level: 3

::: rubband.WindowOption
    options:
      heading_level: 3

::: rubband.SmoothingOption
    options:
      heading_level: 3

::: rubband.FormantOption
    options:
      heading_level: 3

::: rubband.PitchOption
    options:
      heading_level: 3

::: rubband.ChannelsOption
    options:
      heading_level: 3

::: rubband.EngineOption
    options:
      heading_level: 3

::: rubband.PresetOption
    options:
      heading_level: 3
