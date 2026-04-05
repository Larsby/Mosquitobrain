#include "PluginEditor.h"

MosquitobrainAudioProcessorEditor::MosquitobrainAudioProcessorEditor(MosquitobrainAudioProcessor &p)
    : juce::AudioProcessorEditor(&p), audioProcessor(p)
{
    setSize(780, 700);

    titleLabel.setText("Mosquitobrain - Neural Network", juce::dontSendNotification);
    titleLabel.setJustificationType(juce::Justification::centredLeft);
    titleLabel.setFont(juce::FontOptions(20.0f, juce::Font::bold));
    addAndMakeVisible(titleLabel);

    playbackSectionLabel.setText("Playback Controls", juce::dontSendNotification);
    playbackSectionLabel.setFont(juce::FontOptions(15.0f, juce::Font::bold));
    addAndMakeVisible(playbackSectionLabel);

    playbackDryWetLabel.setText("Dry / Wet", juce::dontSendNotification);
    playbackDryWetLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(playbackDryWetLabel);

    dryWetSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    dryWetSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 70, 20);
    dryWetSlider.setName("Dry/Wet");
    addAndMakeVisible(dryWetSlider);

    auto setupLinearSlider = [this](juce::Slider &s, const juce::String &name)
    {
        s.setSliderStyle(juce::Slider::LinearHorizontal);
        s.setTextBoxStyle(juce::Slider::TextBoxRight, false, 56, 20);
        s.setName(name);
        addAndMakeVisible(s);
    };

    setupLinearSlider(nnComplexitySlider, "NN Complexity");
    setupLinearSlider(nnDepthSlider, "NN Depth");
    setupLinearSlider(nnResidualSlider, "NN Residual");
    setupLinearSlider(nnSaturationSlider, "NN Saturation");
    setupLinearSlider(nnRecurrenceSlider, "NN Recurrence");
    setupLinearSlider(pastContextSamplesSlider, "Past Context Samples");
    setupLinearSlider(futureContextSamplesSlider, "Future Context Samples");

    auto setupNNLabel = [this](juce::Label &lbl, const juce::String &text)
    {
        lbl.setText(text, juce::dontSendNotification);
        lbl.setFont(juce::FontOptions(11.0f));
        lbl.setJustificationType(juce::Justification::centredRight);
        addAndMakeVisible(lbl);
    };
    setupNNLabel(nnComplexityLabel, "Complexity");
    setupNNLabel(nnDepthLabel, "Depth");
    setupNNLabel(nnResidualLabel, "Residual");
    setupNNLabel(nnSaturationLabel, "Saturation");
    setupNNLabel(nnRecurrenceLabel, "Recurrence");
    setupNNLabel(pastContextSamplesLabel, "Past Samples");
    setupNNLabel(futureContextSamplesLabel, "Future Samples");

    // NN arch sliders are training-time settings only — no APVTS attachment.
    nnComplexitySlider.setRange(0.0, 1.0, 0.001);
    nnDepthSlider.setRange(1.0, 6.0, 1.0);
    nnResidualSlider.setRange(0.0, 1.0, 0.001);
    nnSaturationSlider.setRange(0.2, 2.5, 0.001);
    nnRecurrenceSlider.setRange(0.0, 1.0, 0.001);
    pastContextSamplesSlider.setRange(0.0, 40.0, 1.0);
    futureContextSamplesSlider.setRange(0.0, 40.0, 1.0);

    dryWetAttachment = std::make_unique<SliderAttachment>(audioProcessor.parameters, "dry_wet", dryWetSlider);
    trainingAmountAttachment = std::make_unique<SliderAttachment>(audioProcessor.parameters, "train_amount", trainingAmountSlider);

    auto resetPresetOnEdit = [this]
    {
        if (!applyingPreset)
            architecturePresetBox.setSelectedId(1, juce::dontSendNotification);
    };
    nnComplexitySlider.onValueChange = resetPresetOnEdit;
    nnDepthSlider.onValueChange = resetPresetOnEdit;
    nnResidualSlider.onValueChange = resetPresetOnEdit;
    nnSaturationSlider.onValueChange = resetPresetOnEdit;
    nnRecurrenceSlider.onValueChange = resetPresetOnEdit;
    pastContextSamplesSlider.onValueChange = resetPresetOnEdit;
    futureContextSamplesSlider.onValueChange = resetPresetOnEdit;

    trainingSectionLabel.setText("Training From Dry/Wet WAV", juce::dontSendNotification);
    trainingSectionLabel.setFont(juce::FontOptions(15.0f, juce::Font::bold));
    addAndMakeVisible(trainingSectionLabel);

    trainingAmountLabel.setText("Training Amount", juce::dontSendNotification);
    addAndMakeVisible(trainingAmountLabel);

    trainingAmountSlider.setRange(0.0, 1.0, 0.001);
    trainingAmountSlider.setSkewFactorFromMidPoint(0.5);
    trainingAmountSlider.setValue(0.7, juce::dontSendNotification);
    trainingAmountSlider.onValueChange = [this]
    {
        const int passes = juce::jlimit(1, 600, 1 + juce::roundToInt(static_cast<float>(trainingAmountSlider.getValue()) * 599.0f));
        trainingAmountValueLabel.setText(juce::String(passes) + " passes", juce::dontSendNotification);
    };
    addAndMakeVisible(trainingAmountSlider);

    trainingAmountValueLabel.setText("420 passes", juce::dontSendNotification);
    trainingAmountValueLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(trainingAmountValueLabel);

    architecturePresetLabel.setText("Architecture Preset", juce::dontSendNotification);
    addAndMakeVisible(architecturePresetLabel);

    architecturePresetBox.addItem("Custom", 1);
    architecturePresetBox.addItem("Filter", 2);
    architecturePresetBox.addItem("Hybrid", 3);
    architecturePresetBox.addItem("Deep", 4);
    architecturePresetBox.setSelectedId(1, juce::dontSendNotification);
    architecturePresetBox.onChange = [this]
    {
        applyArchitecturePreset(architecturePresetBox.getSelectedId());
    };
    addAndMakeVisible(architecturePresetBox);

    browseDryButton.onClick = [this]
    { browseDryFile(); };
    browseWetButton.onClick = [this]
    { browseWetFile(); };
    trainButton.onClick = [this]
    { triggerTrainingFromFiles(); };
    cancelButton.onClick = [this]
    {
        audioProcessor.cancelTraining();
        statusLabel.setText("Training cancel requested.", juce::dontSendNotification);
    };

    addAndMakeVisible(browseDryButton);
    addAndMakeVisible(browseWetButton);
    addAndMakeVisible(trainButton);
    addAndMakeVisible(cancelButton);

    dryFileLabel.setText("Dry: (none)", juce::dontSendNotification);
    wetFileLabel.setText("Wet: (none)", juce::dontSendNotification);
    addAndMakeVisible(dryFileLabel);
    addAndMakeVisible(wetFileLabel);

    statusLabel.setText("Ready.", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
    addAndMakeVisible(statusLabel);

    progressLabel.setText("Progress: 0% | ETA: --", juce::dontSendNotification);
    progressLabel.setColour(juce::Label::textColourId, juce::Colours::lightgrey);
    addAndMakeVisible(progressLabel);

    loadArchitecture();
    juce::String restoredDryName;
    juce::String restoredWetName;
    int restoredUsableSamples = 0;
    if (audioProcessor.getPersistedTrainingDataInfo(restoredDryName, restoredWetName, restoredUsableSamples))
    {
        dryFileLabel.setText("Dry: " + (restoredDryName.isNotEmpty() ? restoredDryName : "restored snapshot"), juce::dontSendNotification);
        wetFileLabel.setText("Wet: " + (restoredWetName.isNotEmpty() ? restoredWetName : "restored snapshot"), juce::dontSendNotification);
        statusLabel.setText("Restored model and training data.", juce::dontSendNotification);
        progressLabel.setText("Restored usable samples: " + juce::String(restoredUsableSamples), juce::dontSendNotification);
    }
    startTimerHz(8);
}

void MosquitobrainAudioProcessorEditor::applyArchitecturePreset(int presetId)
{
    if (presetId <= 1)
        return;

    applyingPreset = true;

    auto setArchSliderValues = [this](float complexity,
                                      float depth,
                                      float residual,
                                      float saturation,
                                      float recurrence,
                                      float pastSamples,
                                      float futureSamples)
    {
        nnComplexitySlider.setValue(complexity, juce::dontSendNotification);
        nnDepthSlider.setValue(depth, juce::dontSendNotification);
        nnResidualSlider.setValue(residual, juce::dontSendNotification);
        nnSaturationSlider.setValue(saturation, juce::dontSendNotification);
        nnRecurrenceSlider.setValue(recurrence, juce::dontSendNotification);
        pastContextSamplesSlider.setValue(pastSamples, juce::dontSendNotification);
        futureContextSamplesSlider.setValue(futureSamples, juce::dontSendNotification);
    };

    switch (presetId)
    {
    case 2: // Filter
        setArchSliderValues(0.05f, 1.0f, 0.10f, 0.9f, 0.20f, 4.0f, 1.0f);
        break;
    case 3: // Hybrid
        setArchSliderValues(0.50f, 3.0f, 0.35f, 1.2f, 0.40f, 12.0f, 8.0f);
        break;
    case 4: // Deep
        setArchSliderValues(0.95f, 6.0f, 0.55f, 1.8f, 0.65f, 24.0f, 18.0f);
        break;
    default:
        break;
    }

    applyingPreset = false;
    statusLabel.setText("Architecture preset applied.", juce::dontSendNotification);
}

MosquitobrainAudioProcessorEditor::~MosquitobrainAudioProcessorEditor()
{
    stopTimer();

    // Detach listeners while sliders are still alive to avoid host-close crashes.
    trainingAmountAttachment.reset();
    dryWetAttachment.reset();
}

void MosquitobrainAudioProcessorEditor::paint(juce::Graphics &g)
{
    g.fillAll(juce::Colour(0xff0f1720));

    static const juce::File backgroundPath = juce::File(__FILE__)
                                                 .getParentDirectory()
                                                 .getParentDirectory()
                                                 .getChildFile("Design")
                                                 .getChildFile("background.png");
    static juce::Image backgroundImage;
    static juce::int64 backgroundLastWrite = 0;

    const juce::int64 currentLastWrite = backgroundPath.existsAsFile()
                                             ? backgroundPath.getLastModificationTime().toMilliseconds()
                                             : 0;
    if (currentLastWrite != backgroundLastWrite)
    {
        backgroundLastWrite = currentLastWrite;
        backgroundImage = juce::ImageFileFormat::loadFrom(backgroundPath);
    }

    if (backgroundImage.isValid())
        g.drawImageWithin(backgroundImage, 0, 0, getWidth(), getHeight(), juce::RectanglePlacement::fillDestination);

    auto outerArea = getLocalBounds().reduced(14);
    g.setColour(juce::Colour(0xff1b2735).withAlpha(0.82f));
    g.fillRoundedRectangle(outerArea.toFloat(), 10.0f);

    // Mirrors resized() exactly so tinted rectangles align with content.
    auto area = getLocalBounds().reduced(24);
    area.removeFromTop(30); // title
    area.removeFromTop(6);  // gap
    auto playbackRect = area.removeFromTop(54);
    area.removeFromTop(6); // gap
    auto trainingRect = area;

    g.setColour(juce::Colour(0xff223447).withAlpha(0.72f));
    g.fillRoundedRectangle(playbackRect.toFloat(), 8.0f);
    g.fillRoundedRectangle(trainingRect.toFloat(), 8.0f);

    g.setColour(juce::Colour(0xff2fd9b5));
    g.drawRoundedRectangle(playbackRect.toFloat(), 8.0f, 1.0f);
    g.drawRoundedRectangle(trainingRect.toFloat(), 8.0f, 1.0f);
}

void MosquitobrainAudioProcessorEditor::resized()
{
    auto area = getLocalBounds().reduced(24);

    titleLabel.setBounds(area.removeFromTop(30));
    area.removeFromTop(6);

    // ── Playback section ──────────────
    auto playbackArea = area.removeFromTop(54).reduced(10);
    playbackSectionLabel.setBounds(playbackArea.removeFromTop(20));
    playbackArea.removeFromTop(4);

    auto dryWetRow = playbackArea.removeFromTop(26);
    playbackDryWetLabel.setBounds(dryWetRow.removeFromLeft(100));
    dryWetSlider.setBounds(dryWetRow);

    area.removeFromTop(6);

    // ── Training section (includes architecture config + training controls) ──────────────
    auto trainingArea = area.reduced(10);
    trainingSectionLabel.setBounds(trainingArea.removeFromTop(20));
    trainingArea.removeFromTop(6);

    // Architecture preset
    auto presetRow = trainingArea.removeFromTop(26);
    architecturePresetLabel.setBounds(presetRow.removeFromLeft(140));
    architecturePresetBox.setBounds(presetRow.removeFromLeft(150));
    trainingArea.removeFromTop(6);

    // NN architecture sliders
    auto placeNNRow = [&](juce::Label &lbl, juce::Slider &sl)
    {
        trainingArea.removeFromTop(2);
        auto row = trainingArea.removeFromTop(14);
        lbl.setBounds(row.removeFromLeft(90));
        sl.setBounds(row);
    };
    placeNNRow(nnComplexityLabel, nnComplexitySlider);
    placeNNRow(nnDepthLabel, nnDepthSlider);
    placeNNRow(nnResidualLabel, nnResidualSlider);
    placeNNRow(nnSaturationLabel, nnSaturationSlider);
    placeNNRow(nnRecurrenceLabel, nnRecurrenceSlider);
    placeNNRow(pastContextSamplesLabel, pastContextSamplesSlider);
    placeNNRow(futureContextSamplesLabel, futureContextSamplesSlider);
    trainingArea.removeFromTop(10);

    // Training amount
    auto row1 = trainingArea.removeFromTop(26);
    trainingAmountLabel.setBounds(row1.removeFromLeft(120));
    trainingAmountSlider.setBounds(row1.removeFromLeft(190));
    trainingAmountValueLabel.setBounds(row1.removeFromLeft(70));

    trainingArea.removeFromTop(10);

    auto row2 = trainingArea.removeFromTop(26);
    browseDryButton.setBounds(row2.removeFromLeft(170));
    row2.removeFromLeft(8);
    dryFileLabel.setBounds(row2);

    trainingArea.removeFromTop(4);

    auto row3 = trainingArea.removeFromTop(26);
    browseWetButton.setBounds(row3.removeFromLeft(170));
    row3.removeFromLeft(8);
    wetFileLabel.setBounds(row3);

    trainingArea.removeFromTop(6);

    auto row4 = trainingArea.removeFromTop(30);
    trainButton.setBounds(row4.removeFromLeft(120));
    row4.removeFromLeft(8);
    cancelButton.setBounds(row4.removeFromLeft(100));

    trainingArea.removeFromTop(6);
    statusLabel.setBounds(trainingArea.removeFromTop(22));
    trainingArea.removeFromTop(2);
    progressLabel.setBounds(trainingArea.removeFromTop(20));
}

void MosquitobrainAudioProcessorEditor::timerCallback()
{
    updateStatus();
}

void MosquitobrainAudioProcessorEditor::browseDryFile()
{
    fileChooser = std::make_unique<juce::FileChooser>("Select dry input WAV",
                                                      juce::File{},
                                                      "*.wav;*.aif;*.aiff;*.flac");
    fileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                             [this](const juce::FileChooser &chooser)
                             {
                                 const auto selected = chooser.getResult();
                                 if (selected.existsAsFile())
                                 {
                                     dryTrainingFile = selected;
                                     dryFileLabel.setText("Dry: " + selected.getFileName(), juce::dontSendNotification);
                                 }
                             });
}

void MosquitobrainAudioProcessorEditor::browseWetFile()
{
    fileChooser = std::make_unique<juce::FileChooser>("Select wet target WAV",
                                                      juce::File{},
                                                      "*.wav;*.aif;*.aiff;*.flac");
    fileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                             [this](const juce::FileChooser &chooser)
                             {
                                 const auto selected = chooser.getResult();
                                 if (selected.existsAsFile())
                                 {
                                     wetTrainingFile = selected;
                                     wetFileLabel.setText("Wet: " + selected.getFileName(), juce::dontSendNotification);
                                 }
                             });
}

void MosquitobrainAudioProcessorEditor::triggerTrainingFromFiles()
{
    juce::String error;
    const float amount = static_cast<float>(trainingAmountSlider.getValue());
    const bool haveSelectedWaveFiles = dryTrainingFile.existsAsFile() && wetTrainingFile.existsAsFile();

    if (haveSelectedWaveFiles)
    {
        if (audioProcessor.queueTrainingFromWaveFiles(dryTrainingFile,
                                                      wetTrainingFile,
                                                      amount,
                                                      static_cast<float>(nnComplexitySlider.getValue()),
                                                      static_cast<float>(nnDepthSlider.getValue()),
                                                      static_cast<float>(nnResidualSlider.getValue()),
                                                      static_cast<float>(nnSaturationSlider.getValue()),
                                                      static_cast<float>(nnRecurrenceSlider.getValue()),
                                                      static_cast<float>(pastContextSamplesSlider.getValue()),
                                                      static_cast<float>(futureContextSamplesSlider.getValue()),
                                                      error))
        {
            int drySamples = 0;
            int wetBeforeResample = 0;
            int wetAfterResample = 0;
            int usableSamples = 0;
            int estimatedLag = 0;
            audioProcessor.getLastWaveFileLoadStats(drySamples,
                                                    wetBeforeResample,
                                                    wetAfterResample,
                                                    usableSamples,
                                                    estimatedLag);

            statusLabel.setText("Training started.", juce::dontSendNotification);
            progressLabel.setText("Loaded Dry/Wet/Resamp/Usable: " +
                                      juce::String(drySamples) + "/" +
                                      juce::String(wetBeforeResample) + "/" +
                                      juce::String(wetAfterResample) + "/" +
                                      juce::String(usableSamples) +
                                      " | Lag: " + juce::String(estimatedLag) + " smp",
                                  juce::dontSendNotification);
            return;
        }

        statusLabel.setText("Training failed: " + error, juce::dontSendNotification);
        return;
    }

    juce::String restoredDryName;
    juce::String restoredWetName;
    int restoredUsableSamples = 0;
    if (audioProcessor.getPersistedTrainingDataInfo(restoredDryName, restoredWetName, restoredUsableSamples) &&
        audioProcessor.retrainLastCapture())
    {
        statusLabel.setText("Training started from restored snapshot.", juce::dontSendNotification);
        progressLabel.setText("Restored Dry/Wet/Usable: " +
                                  (restoredDryName.isNotEmpty() ? restoredDryName : juce::String("snapshot")) + "/" +
                                  (restoredWetName.isNotEmpty() ? restoredWetName : juce::String("snapshot")) + "/" +
                                  juce::String(restoredUsableSamples),
                              juce::dontSendNotification);
        return;
    }

    statusLabel.setText("Training failed: choose dry/wet WAVs or restore a saved snapshot.", juce::dontSendNotification);
}

void MosquitobrainAudioProcessorEditor::updateStatus()
{
    const bool training = audioProcessor.isTrainingInProgress();
    const float progress = audioProcessor.getTrainingProgress();
    const float etaSeconds = audioProcessor.getEstimatedTrainingSecondsRemaining();
    float realtimeCpu = 0.0f;
    int latencySamples = 0;
    int pastCtx = 0;
    int futureCtx = 0;
    audioProcessor.getRealtimeDiagnostics(realtimeCpu, latencySamples, pastCtx, futureCtx);
    float mse = 0.0f;
    float accuracy1 = 0.0f;
    float accuracy5 = 0.0f;
    audioProcessor.getTrainingMetrics(mse, accuracy1, accuracy5);
    trainButton.setEnabled(!training);
    cancelButton.setEnabled(training);

    if (training)
    {
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
        statusLabel.setText("Training in progress...", juce::dontSendNotification);

        const int percent = juce::roundToInt(progress * 100.0f);
        const juce::String acc1Text = juce::String(juce::roundToInt(accuracy1 * 100.0f)) + "%";
        const juce::String acc5Text = juce::String(juce::roundToInt(accuracy5 * 100.0f)) + "%";
        juce::String etaText = "--";
        if (etaSeconds > 0.05f)
            etaText = juce::String(etaSeconds, 1) + "s";
        progressLabel.setText("Progress: " + juce::String(percent) + "% | ETA: " + etaText +
                                  " | Loss: " + juce::String(mse, 6) +
                                  " | A@1%: " + acc1Text + " | A@5%: " + acc5Text,
                              juce::dontSendNotification);
    }
    else if (statusLabel.getText().startsWith("Training in progress"))
    {
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);
        statusLabel.setText("Training idle.", juce::dontSendNotification);
        progressLabel.setText("Runtime: CPU " + juce::String(realtimeCpu, 1) + "% | Latency " + juce::String(latencySamples) +
                                  " smp | Context P/F " + juce::String(pastCtx) + "/" + juce::String(futureCtx),
                              juce::dontSendNotification);
    }

    bool completedSuccess = false;
    if (audioProcessor.consumeLastTrainingCompletion(completedSuccess))
    {
        const juce::String msg = completedSuccess
                                     ? "Training complete."
                                     : "Training failed or cancelled.";

        statusLabel.setColour(juce::Label::textColourId,
                              completedSuccess ? juce::Colours::lightgreen : juce::Colours::orange);
        statusLabel.setText(msg, juce::dontSendNotification);
        if (completedSuccess)
        {
            const juce::String acc1Text = juce::String(juce::roundToInt(accuracy1 * 100.0f)) + "%";
            const juce::String acc5Text = juce::String(juce::roundToInt(accuracy5 * 100.0f)) + "%";
            progressLabel.setText("Progress: 100% | ETA: done | Loss: " + juce::String(mse, 6) +
                                      " | A@1%: " + acc1Text + " | A@5%: " + acc5Text,
                                  juce::dontSendNotification);
        }
        else
        {
            progressLabel.setText("Progress: -- | Cancelled",
                                  juce::dontSendNotification);
        }
    }

    if (!training && !statusLabel.getText().startsWith("Training complete") && !statusLabel.getText().startsWith("Training failed"))
    {
        progressLabel.setText("Runtime: CPU " + juce::String(realtimeCpu, 1) + "% | Latency " + juce::String(latencySamples) +
                                  " smp | Context P/F " + juce::String(pastCtx) + "/" + juce::String(futureCtx),
                              juce::dontSendNotification);
    }
}

void MosquitobrainAudioProcessorEditor::loadArchitecture()
{
    float complexity, depth, residual, saturation, recurrence, pastSamples, futureSamples;
    audioProcessor.getModelArchitecture(complexity, depth, residual, saturation, recurrence, pastSamples, futureSamples);

    applyingPreset = true;
    nnComplexitySlider.setValue(complexity, juce::dontSendNotification);
    nnDepthSlider.setValue(depth, juce::dontSendNotification);
    nnResidualSlider.setValue(residual, juce::dontSendNotification);
    nnSaturationSlider.setValue(saturation, juce::dontSendNotification);
    nnRecurrenceSlider.setValue(recurrence, juce::dontSendNotification);
    pastContextSamplesSlider.setValue(pastSamples, juce::dontSendNotification);
    futureContextSamplesSlider.setValue(futureSamples, juce::dontSendNotification);
    applyingPreset = false;
}
