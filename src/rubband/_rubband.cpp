#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <rubberband/RubberBandStretcher.h>

namespace nb = nanobind;

using AudioArray = nb::ndarray<nb::numpy, const float, nb::shape<-1, -1>, nb::c_contig>;
using OutputArray = nb::ndarray<nb::numpy, float, nb::shape<-1, -1>, nb::c_contig>;

OutputArray stretch_float32(
    AudioArray audio,
    int sample_rate,
    double time_ratio,
    double pitch_scale
) {
    const size_t frames = audio.shape(0);
    const size_t channels = audio.shape(1);
    const float *input = audio.data();

    std::vector<std::vector<float>> planar(channels, std::vector<float>(frames));
    for (size_t frame = 0; frame < frames; ++frame) {
        for (size_t channel = 0; channel < channels; ++channel) {
            planar[channel][frame] = input[frame * channels + channel];
        }
    }

    std::vector<const float *> input_channels(channels);
    for (size_t channel = 0; channel < channels; ++channel) {
        input_channels[channel] = planar[channel].data();
    }

    RubberBand::RubberBandStretcher stretcher(
        static_cast<size_t>(sample_rate),
        channels,
        RubberBand::RubberBandStretcher::OptionProcessOffline |
            RubberBand::RubberBandStretcher::OptionThreadingNever,
        time_ratio,
        pitch_scale
    );
    stretcher.setExpectedInputDuration(frames);

    nb::gil_scoped_release release;
    stretcher.study(input_channels.data(), frames, true);
    stretcher.process(input_channels.data(), frames, true);

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
    nb::gil_scoped_acquire acquire;

    auto output = std::make_unique<std::vector<float>>(retrieved * channels);
    for (size_t frame = 0; frame < retrieved; ++frame) {
        for (size_t channel = 0; channel < channels; ++channel) {
            (*output)[frame * channels + channel] = output_channels[channel][frame];
        }
    }

    float *data = output->data();
    nb::capsule owner(output.release(), [](void *p) noexcept {
        delete static_cast<std::vector<float> *>(p);
    });
    return OutputArray(data, {retrieved, channels}, owner);
}

NB_MODULE(_rubband, module) {
    module.def(
        "stretch_float32",
        &stretch_float32,
        nb::arg("audio"),
        nb::arg("sample_rate"),
        nb::arg("time_ratio"),
        nb::arg("pitch_scale")
    );
}
