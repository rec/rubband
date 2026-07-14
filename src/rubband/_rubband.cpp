#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <rubberband/RubberBandStretcher.h>

namespace nb = nanobind;

using AudioArray = nb::ndarray<nb::numpy, const float, nb::shape<-1, -1>, nb::f_contig>;
using OutputArray = nb::ndarray<nb::numpy, float, nb::shape<-1, -1>, nb::f_contig>;

std::vector<std::vector<float>> copy_planar(AudioArray audio, size_t expected_channels) {
    const size_t frames = audio.shape(0);
    const size_t channels = audio.shape(1);
    if (frames < 1) {
        throw std::runtime_error("expected at least one frame");
    }
    if (channels != expected_channels) {
        throw std::runtime_error("audio channel count does not match stretcher");
    }
    if (audio.stride(0) != 1) {
        throw std::runtime_error("expected Fortran-order audio");
    }
    const float *input = audio.data();
    std::vector<std::vector<float>> input_storage(channels);
    for (size_t channel = 0; channel < channels; ++channel) {
        input_storage[channel].assign(
            input + channel * frames,
            input + (channel + 1) * frames
        );
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
    size_t channels
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
        std::copy_n(
            output_channels[channel].data(),
            retrieved,
            output->data() + channel * retrieved
        );
    }

    float *data = output->data();
    nb::capsule owner(output.release(), [](void *p) noexcept {
        delete static_cast<std::vector<float> *>(p);
    });
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
        stretcher_.study(input_channels.data(), audio.shape(0), final);
    }

    void process(AudioArray audio, bool final) {
        auto input_storage = copy_planar(audio, channels_);
        auto input_channels = channel_pointers(input_storage);
        nb::gil_scoped_release release;
        stretcher_.process(input_channels.data(), audio.shape(0), final);
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
    const size_t frames = audio.shape(0);
    const size_t channels = audio.shape(1);
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

    return retrieve_from(stretcher, channels);
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
        .def("study", &Stretcher::study, nb::arg("audio"), nb::arg("final") = false)
        .def("process", &Stretcher::process, nb::arg("audio"), nb::arg("final") = false)
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
        nb::arg("audio"),
        nb::arg("sample_rate"),
        nb::arg("time_ratio"),
        nb::arg("pitch_scale"),
        nb::arg("option_flags")
    );
}
