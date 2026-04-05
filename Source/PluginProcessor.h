#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

class MosquitobrainAudioProcessor : public juce::AudioProcessor
{
public:
    MosquitobrainAudioProcessor();
    ~MosquitobrainAudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout &layouts) const override;

    void processBlock(juce::AudioBuffer<float> &, juce::MidiBuffer &) override;
    using AudioProcessor::processBlock;

    juce::AudioProcessorEditor *createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String &newName) override;

    void getStateInformation(juce::MemoryBlock &destData) override;
    void setStateInformation(const void *data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState parameters;

    void startSourceCapture() noexcept;
    void startTargetCapture() noexcept;
    void stopCapture() noexcept;
    bool queueTrainingFromCapturedAudio();
    bool queueTrainingFromWaveFiles(const juce::File &dryFile,
                                    const juce::File &wetFile,
                                    float trainAmount,
                                    float nnComplexity,
                                    float nnDepth,
                                    float nnResidual,
                                    float nnSaturation,
                                    float nnRecurrence,
                                    float pastContextSamples,
                                    float futureContextSamples,
                                    juce::String &errorMessage);
    bool retrainLastCapture();
    void cancelTraining() noexcept;
    bool isTrainingInProgress() const noexcept;
    bool isCaptureActive() const noexcept;
    float getTrainingProgress() const noexcept;
    float getEstimatedTrainingSecondsRemaining() const noexcept;
    void getTrainingMetrics(float &outMse, float &outAccuracy1, float &outAccuracy5) const noexcept;
    void getLastWaveFileLoadStats(int &outDrySamples,
                                  int &outWetSamplesBeforeResample,
                                  int &outWetSamplesAfterResample,
                                  int &outUsableSamples,
                                  int &outEstimatedLagSamples) const noexcept;
    bool getPersistedTrainingDataInfo(juce::String &outDryName,
                                      juce::String &outWetName,
                                      int &outUsableSamples) const;
    void getRealtimeDiagnostics(float &outCpuPercent,
                                int &outLatencySamples,
                                int &outPastContextSamples,
                                int &outFutureContextSamples) const noexcept;
    bool consumeLastTrainingCompletion(bool &outSuccess) noexcept;
    void getModelArchitecture(float &complexity, float &depth, float &residual,
                              float &saturation, float &recurrence,
                              float &pastContextSamples,
                              float &futureContextSamples) const noexcept;

private:
    struct NeuralModel
    {
        float drive = 1.2f;
        float recurrentA = 0.45f;
        float recurrentB = 0.2f;
        float bias = 0.0f;
        float outputBlend = 0.72f;
        float nnComplexity = 0.2f;
        float nnDepth = 2.0f;
        float nnResidual = 0.25f;
        float nnSaturation = 1.0f;
        float nnRecurrence = 0.35f;
        float pastContextSamples = 0.0f;
        float futureContextSamples = 0.0f;
    };

    struct TrainingSnapshot
    {
        std::vector<float> source;
        std::vector<float> target;

        float trainAmount = 0.7f;
        // Architecture baked in at training time (not live playback params).
        float nnComplexity = 0.2f;
        float nnDepth = 2.0f;
        float nnResidual = 0.25f;
        float nnSaturation = 1.0f;
        float nnRecurrence = 0.35f;
        float pastContextSamples = 0.0f;
        float futureContextSamples = 0.0f;
    };

    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();
    void initialiseDefaultModels();

    float runInferenceSample(float inputSample,
                             float contextPast,
                             float contextFuture,
                             int channel) noexcept;

    void captureSample(float sample) noexcept;
    bool makeTrainingSnapshot(TrainingSnapshot &snapshot) const;
    bool loadMonoSamplesFromWaveFile(const juce::File &file,
                                     std::vector<float> &outSamples,
                                     double &outSampleRate,
                                     juce::String &errorMessage) const;
    void clearPersistedTrainingData();
    void trainingWorkerLoop();
    bool runTrainingPass(const TrainingSnapshot &snapshot);

    static constexpr int kCaptureNone = 0;
    static constexpr int kCaptureSource = 1;
    static constexpr int kCaptureTarget = 2;

    static constexpr int kMaxContextSize = 40;
    static constexpr int kContextRingSize = (2 * kMaxContextSize) + 1;

    std::array<std::array<float, 2>, 2> inferenceState{{{0.0f, 0.0f}, {0.0f, 0.0f}}};
    std::array<std::array<float, kContextRingSize>, 2> contextBuffer{};
    std::array<int, 2> contextWritePos{};
    std::array<int, 2> contextSampleCount{};
    NeuralModel activeModel;
    NeuralModel pendingModel;

    juce::LinearSmoothedValue<float> dryWetSmoothed;
    juce::AudioBuffer<float> dryBuffer;
    juce::AudioBuffer<float> sourceCaptureBuffer;
    juce::AudioBuffer<float> targetCaptureBuffer;

    std::atomic<int> captureMode{kCaptureNone};
    std::atomic<int> sourceWritePosition{0};
    std::atomic<int> targetWritePosition{0};
    std::atomic<float> latestTrainAmount{0.7f};
    std::atomic<bool> modelUpdatePending{false};
    std::atomic<bool> workerStopRequested{false};
    std::atomic<bool> trainingJobPending{false};
    std::atomic<bool> trainingCancelRequested{false};
    std::atomic<bool> trainingInProgress{false};
    std::atomic<float> trainingProgress{0.0f};
    std::atomic<float> trainingSecondsRemaining{0.0f};
    std::atomic<float> trainingMse{0.0f};
    std::atomic<float> trainingAccuracy1{0.0f};
    std::atomic<float> trainingAccuracy5{0.0f};
    std::atomic<float> realtimeCpuPercent{0.0f};
    std::atomic<int> lastDrySamples{0};
    std::atomic<int> lastWetSamplesBeforeResample{0};
    std::atomic<int> lastWetSamplesAfterResample{0};
    std::atomic<int> lastUsableSamples{0};
    std::atomic<int> lastEstimatedLagSamples{0};

    std::atomic<int> lastCompletedTrainingSuccess{0};
    std::atomic<bool> completionPendingForUi{false};

    mutable std::mutex trainingMutex;
    TrainingSnapshot pendingTrainingSnapshot;
    TrainingSnapshot lastTrainingSnapshot;
    juce::String lastDryTrainingName;
    juce::String lastWetTrainingName;
    bool hasPendingSnapshot = false;
    bool hasLastSnapshot = false;
    std::thread trainingThread;

    int maxCaptureSamples = 0;

    double currentSampleRate = 44100.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MosquitobrainAudioProcessor)
};
