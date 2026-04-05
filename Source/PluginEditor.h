#pragma once

#include <memory>

#include <juce_gui_basics/juce_gui_basics.h>

#include "PluginProcessor.h"

class MosquitobrainAudioProcessorEditor : public juce::AudioProcessorEditor,
                                          private juce::Timer
{
public:
    explicit MosquitobrainAudioProcessorEditor(MosquitobrainAudioProcessor &);
    ~MosquitobrainAudioProcessorEditor() override;

    void paint(juce::Graphics &) override;
    void resized() override;

private:
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;

    void timerCallback() override;
    void browseDryFile();
    void browseWetFile();
    void triggerTrainingFromFiles();
    void updateStatus();
    void applyArchitecturePreset(int presetId);

    MosquitobrainAudioProcessor &audioProcessor;

    juce::Label titleLabel;

    juce::Label playbackSectionLabel;
    juce::Label architecturePresetLabel;
    juce::ComboBox architecturePresetBox;
    juce::Slider dryWetSlider;
    juce::Slider nnComplexitySlider;
    juce::Slider nnDepthSlider;
    juce::Slider nnResidualSlider;
    juce::Slider nnSaturationSlider;
    juce::Slider nnRecurrenceSlider;
    void loadArchitecture();
    juce::Label playbackDryWetLabel;

    juce::Label nnComplexityLabel, nnDepthLabel, nnResidualLabel, nnSaturationLabel, nnRecurrenceLabel;
    juce::Slider pastContextSamplesSlider;
    juce::Slider futureContextSamplesSlider;
    juce::Label pastContextSamplesLabel;
    juce::Label futureContextSamplesLabel;
    bool applyingPreset{false};

    juce::Label trainingSectionLabel;
    juce::Slider trainingAmountSlider;
    juce::Label trainingAmountLabel;
    juce::Label trainingAmountValueLabel;

    juce::TextButton browseDryButton{"Choose Dry WAV"};
    juce::TextButton browseWetButton{"Choose Wet WAV"};
    juce::TextButton trainButton{"Train"};
    juce::TextButton cancelButton{"Cancel"};

    juce::Label dryFileLabel;
    juce::Label wetFileLabel;
    juce::Label statusLabel;
    juce::Label progressLabel;

    // Must be declared after sliders so they are destroyed first (safe listener detach).
    std::unique_ptr<SliderAttachment> dryWetAttachment;
    std::unique_ptr<SliderAttachment> trainingAmountAttachment;

    juce::File dryTrainingFile;
    juce::File wetTrainingFile;
    std::unique_ptr<juce::FileChooser> fileChooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MosquitobrainAudioProcessorEditor)
};
