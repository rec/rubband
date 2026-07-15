#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <rubberband/RubberBandStretcher.h>

namespace nb = nanobind;

using AudioArray = nb::ndarray<const float, nb::device::cpu, nb::c_contig>;
using OutputArray = nb::ndarray<nb::memview, float, nb::device::cpu, nb::c_contig>;

size_t frame_count(AudioArray audio) {
    if (audio.ndim() != 1 && audio.ndim() != 2) {
        throw std::runtime_error("audio must have shape (frames,) or (frames, channels)");
    }
    return audio.shape(0);
}

size_t channel_count(AudioArray audio) {
    if (audio.ndim() == 1) {
        return 1;
    }
    return audio.shape(1);
}

std::vector<std::vector<float>> copy_planar(AudioArray audio, size_t expected_channels) {
    const size_t frames = frame_count(audio);
    const size_t channels = channel_count(audio);
    if (frames < 1) {
        throw std::runtime_error("expected at least one frame");
    }
    if (channels < 1) {
        throw std::runtime_error("expected at least one channel");
    }
    if (channels != expected_channels) {
        throw std::runtime_error("audio channel count does not match stretcher");
    }
    const float *input = audio.data();
    std::vector<std::vector<float>> input_storage(channels);
    for (size_t channel = 0; channel < channels; ++channel) {
        input_storage[channel].resize(frames);
        for (size_t frame = 0; frame < frames; ++frame) {
            input_storage[channel][frame] = input[frame * channels + channel];
        }
    }
    return input_storage;
}

std::vector<const float *> channel_pointers(
    const std::vector<std::vector<float>> &input_storage
) {
    std::vector<const float *> input_channels(input_storage.size());
    for (size_t channel = 0; channel < input_storage.size(); ++channel) {
        input_channels[channel] = input_storage[channel].data();
    }
    return input_channels;
}

OutputArray retrieve_from(
    RubberBand::RubberBandStretcher &stretcher,
    size_t channels,
    bool mono = false
) {
    int available = stretcher.available();
    if (available < 0) {
        throw std::runtime_error("Rubber Band reported negative available output");
    }

    std::vector<std::vector<float>> output_channels(
        channels,
        std::vector<float>(static_cast<size_t>(available))
    );
    std::vector<float *> output_pointers(channels);
    for (size_t channel = 0; channel < channels; ++channel) {
        output_pointers[channel] = output_channels[channel].data();
    }

    size_t retrieved = stretcher.retrieve(
        output_pointers.data(),
        static_cast<size_t>(available)
    );

    auto output = std::make_unique<std::vector<float>>(retrieved * channels);
    for (size_t channel = 0; channel < channels; ++channel) {
        for (size_t frame = 0; frame < retrieved; ++frame) {
            (*output)[frame * channels + channel] = output_channels[channel][frame];
        }
    }

    float *data = output->data();
    nb::capsule owner(output.release(), [](void *p) noexcept {
        delete static_cast<std::vector<float> *>(p);
    });
    if (mono) {
        return OutputArray(data, {retrieved}, owner);
    }
    return OutputArray(data, {retrieved, channels}, owner);
}

class Stretcher {
public:
    Stretcher(
        int sample_rate,
        int channels,
        double time_ratio,
        double pitch_scale,
        int option_flags
    ) :
        channels_(checked_channels(channels)),
        stretcher_(
            static_cast<size_t>(sample_rate),
            channels_,
            option_flags,
            time_ratio,
            pitch_scale
        ) { }

    static size_t checked_channels(int channels) {
        if (channels < 1 || channels > 256) {
            throw std::runtime_error("expected channels between 1 and 256");
        }
        return static_cast<size_t>(channels);
    }

    void study(AudioArray audio, bool final) {
        auto input_storage = copy_planar(audio, channels_);
        auto input_channels = channel_pointers(input_storage);
        nb::gil_scoped_release release;
        stretcher_.study(input_channels.data(), frame_count(audio), final);
    }

    void process(AudioArray audio, bool final) {
        auto input_storage = copy_planar(audio, channels_);
        auto input_channels = channel_pointers(input_storage);
        nb::gil_scoped_release release;
        stretcher_.process(input_channels.data(), frame_count(audio), final);
    }

    void reset() {
        stretcher_.reset();
    }

    void set_time_ratio(double ratio) {
        stretcher_.setTimeRatio(ratio);
    }

    void set_pitch_scale(double scale) {
        stretcher_.setPitchScale(scale);
    }

    void set_formant_scale(double scale) {
        stretcher_.setFormantScale(scale);
    }

    void set_transients_option(int options) {
        stretcher_.setTransientsOption(options);
    }

    void set_detector_option(int options) {
        stretcher_.setDetectorOption(options);
    }

    void set_phase_option(int options) {
        stretcher_.setPhaseOption(options);
    }

    void set_formant_option(int options) {
        stretcher_.setFormantOption(options);
    }

    void set_pitch_option(int options) {
        stretcher_.setPitchOption(options);
    }

    double get_time_ratio() const {
        return stretcher_.getTimeRatio();
    }

    double get_pitch_scale() const {
        return stretcher_.getPitchScale();
    }

    double get_formant_scale() const {
        return stretcher_.getFormantScale();
    }

    size_t get_preferred_start_pad() const {
        return stretcher_.getPreferredStartPad();
    }

    size_t get_start_delay() const {
        return stretcher_.getStartDelay();
    }

    size_t get_latency() const {
        return stretcher_.getLatency();
    }

    size_t get_channel_count() const {
        return stretcher_.getChannelCount();
    }

    void set_expected_input_duration(size_t samples) {
        stretcher_.setExpectedInputDuration(samples);
    }

    void set_max_process_size(size_t samples) {
        stretcher_.setMaxProcessSize(samples);
    }

    size_t get_process_size_limit() const {
        return stretcher_.getProcessSizeLimit();
    }

    size_t get_samples_required() const {
        return stretcher_.getSamplesRequired();
    }

    int available() const {
        return stretcher_.available();
    }

    OutputArray retrieve() {
        return retrieve_from(stretcher_, channels_);
    }

private:
    size_t channels_;
    RubberBand::RubberBandStretcher stretcher_;
};

nb::dict metadata(
    int sample_rate,
    int channels,
    double time_ratio,
    double pitch_scale,
    int option_flags
) {
    if (channels < 1 || channels > 256) {
        throw std::runtime_error("expected channels between 1 and 256");
    }
    RubberBand::RubberBandStretcher stretcher(
        static_cast<size_t>(sample_rate),
        static_cast<size_t>(channels),
        option_flags,
        time_ratio,
        pitch_scale
    );
    nb::dict result;
    result["engine_version"] = stretcher.getEngineVersion();
    result["available"] = stretcher.available();
    result["preferred_start_pad"] = stretcher.getPreferredStartPad();
    result["start_delay"] = stretcher.getStartDelay();
    result["time_ratio"] = stretcher.getTimeRatio();
    result["pitch_scale"] = stretcher.getPitchScale();
    return result;
}

OutputArray stretch_float32(
    AudioArray audio,
    int sample_rate,
    double time_ratio,
    double pitch_scale,
    int option_flags
) {
    const size_t frames = frame_count(audio);
    const size_t channels = channel_count(audio);
    if (channels < 1 || channels > 256) {
        throw std::runtime_error("expected audio shape (frames, channels)");
    }
    auto input_storage = copy_planar(audio, channels);
    auto input_channels = channel_pointers(input_storage);

    RubberBand::RubberBandStretcher stretcher(
        static_cast<size_t>(sample_rate),
        channels,
        option_flags,
        time_ratio,
        pitch_scale
    );
    stretcher.setExpectedInputDuration(frames);

    {
        nb::gil_scoped_release release;
        if ((option_flags & RubberBand::RubberBandStretcher::OptionProcessRealTime) == 0) {
            stretcher.study(input_channels.data(), frames, true);
        }
        stretcher.process(input_channels.data(), frames, true);
    }

    return retrieve_from(stretcher, channels, audio.ndim() == 1);
}

NB_MODULE(_rubband, module) {
    nb::class_<Stretcher>(module, "Stretcher")
        .def(
            nb::init<int, int, double, double, int>(),
            nb::arg("sample_rate"),
            nb::arg("channels"),
            nb::arg("time_ratio"),
            nb::arg("pitch_scale"),
            nb::arg("option_flags")
        )
        .def(
            "study",
            &Stretcher::study,
            nb::arg("audio").noconvert(),
            nb::arg("final") = false
        )
        .def(
            "process",
            &Stretcher::process,
            nb::arg("audio").noconvert(),
            nb::arg("final") = false
        )
        .def("reset", &Stretcher::reset)
        .def("set_time_ratio", &Stretcher::set_time_ratio, nb::arg("ratio"))
        .def("set_pitch_scale", &Stretcher::set_pitch_scale, nb::arg("scale"))
        .def("set_formant_scale", &Stretcher::set_formant_scale, nb::arg("scale"))
        .def(
            "set_transients_option",
            &Stretcher::set_transients_option,
            nb::arg("options")
        )
        .def("set_detector_option", &Stretcher::set_detector_option, nb::arg("options"))
        .def("set_phase_option", &Stretcher::set_phase_option, nb::arg("options"))
        .def("set_formant_option", &Stretcher::set_formant_option, nb::arg("options"))
        .def("set_pitch_option", &Stretcher::set_pitch_option, nb::arg("options"))
        .def("get_time_ratio", &Stretcher::get_time_ratio)
        .def("get_pitch_scale", &Stretcher::get_pitch_scale)
        .def("get_formant_scale", &Stretcher::get_formant_scale)
        .def("get_preferred_start_pad", &Stretcher::get_preferred_start_pad)
        .def("get_start_delay", &Stretcher::get_start_delay)
        .def("get_latency", &Stretcher::get_latency)
        .def("get_channel_count", &Stretcher::get_channel_count)
        .def(
            "set_expected_input_duration",
            &Stretcher::set_expected_input_duration,
            nb::arg("samples")
        )
        .def(
            "set_max_process_size",
            &Stretcher::set_max_process_size,
            nb::arg("samples")
        )
        .def("get_process_size_limit", &Stretcher::get_process_size_limit)
        .def("get_samples_required", &Stretcher::get_samples_required)
        .def("available", &Stretcher::available)
        .def("retrieve", &Stretcher::retrieve);

    module.def(
        "metadata",
        &metadata,
        nb::arg("sample_rate"),
        nb::arg("channels"),
        nb::arg("time_ratio"),
        nb::arg("pitch_scale"),
        nb::arg("option_flags")
    );
    module.def(
        "stretch_float32",
        &stretch_float32,
        nb::arg("audio").noconvert(),
        nb::arg("sample_rate"),
        nb::arg("time_ratio"),
        nb::arg("pitch_scale"),
        nb::arg("option_flags")
    );
}
