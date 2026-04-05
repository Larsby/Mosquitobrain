#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

#include <juce_audio_formats/juce_audio_formats.h>

namespace
{
    constexpr int kMaxPersistedTrainingSamples = 2 * 60 * 48000;

    std::vector<float> resampleLinear(const std::vector<float> &input,
                                      double inputRate,
                                      double outputRate)
    {
        if (input.empty() || inputRate <= 0.0 || outputRate <= 0.0)
            return input;

        if (std::abs(inputRate - outputRate) < 1.0)
            return input;

        const double ratio = outputRate / inputRate;
        const int outCount = juce::jmax(1, static_cast<int>(std::llround(static_cast<double>(input.size()) * ratio)));

        std::vector<float> output(static_cast<size_t>(outCount), 0.0f);

        for (int i = 0; i < outCount; ++i)
        {
            const double sourcePos = static_cast<double>(i) / ratio;
            const int left = juce::jlimit(0, static_cast<int>(input.size()) - 1, static_cast<int>(sourcePos));
            const int right = juce::jlimit(0, static_cast<int>(input.size()) - 1, left + 1);
            const float frac = static_cast<float>(sourcePos - static_cast<double>(left));
            output[static_cast<size_t>(i)] = input[static_cast<size_t>(left)] * (1.0f - frac) + input[static_cast<size_t>(right)] * frac;
        }

        return output;
    }

    int estimateBestLagSamples(const std::vector<float> &dry,
                               const std::vector<float> &wet,
                               int maxLag,
                               int analysisLength)
    {
        if (dry.empty() || wet.empty())
            return 0;

        const int drySize = static_cast<int>(dry.size());
        const int wetSize = static_cast<int>(wet.size());
        const int usable = juce::jmin(analysisLength, drySize, wetSize);
        if (usable < 256)
            return 0;

        double bestScore = -std::numeric_limits<double>::infinity();
        int bestLag = 0;

        for (int lag = -maxLag; lag <= maxLag; ++lag)
        {
            const int dryStart = juce::jmax(0, lag);
            const int wetStart = juce::jmax(0, -lag);
            const int count = juce::jmin(usable - dryStart, usable - wetStart);
            if (count < 256)
                continue;

            double dot = 0.0;
            double dryEnergy = 0.0;
            double wetEnergy = 0.0;

            for (int i = 0; i < count; ++i)
            {
                const float d = dry[static_cast<size_t>(dryStart + i)];
                const float w = wet[static_cast<size_t>(wetStart + i)];
                dot += static_cast<double>(d) * static_cast<double>(w);
                dryEnergy += static_cast<double>(d) * static_cast<double>(d);
                wetEnergy += static_cast<double>(w) * static_cast<double>(w);
            }

            const double denom = std::sqrt(dryEnergy * wetEnergy) + 1.0e-12;
            const double score = dot / denom;
            if (score > bestScore)
            {
                bestScore = score;
                bestLag = lag;
            }
        }

        return bestLag;
    }

    void applyLagAlignment(std::vector<float> &dry,
                           std::vector<float> &wet,
                           int lag)
    {
        if (dry.empty() || wet.empty())
            return;

        if (lag > 0)
        {
            const size_t shift = static_cast<size_t>(juce::jmin(lag, static_cast<int>(dry.size())));
            dry.erase(dry.begin(), dry.begin() + static_cast<std::ptrdiff_t>(shift));
        }
        else if (lag < 0)
        {
            const size_t shift = static_cast<size_t>(juce::jmin(-lag, static_cast<int>(wet.size())));
            wet.erase(wet.begin(), wet.begin() + static_cast<std::ptrdiff_t>(shift));
        }

        const size_t common = juce::jmin(dry.size(), wet.size());
        dry.resize(common);
        wet.resize(common);
    }

    int16_t floatToPcm16(float sample) noexcept
    {
        const float clamped = juce::jlimit(-1.0f, 1.0f, sample);
        return static_cast<int16_t>(juce::roundToInt(clamped * 32767.0f));
    }

    float pcm16ToFloat(int16_t sample) noexcept
    {
        const float scaled = static_cast<float>(sample) / 32768.0f;
        return juce::jlimit(-1.0f, 1.0f, scaled);
    }
}

MosquitobrainAudioProcessor::MosquitobrainAudioProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, juce::Identifier("Mosquitobrain"), createParameterLayout())
{
    initialiseDefaultModels();
    trainingThread = std::thread([this]
                                 { trainingWorkerLoop(); });
}

MosquitobrainAudioProcessor::~MosquitobrainAudioProcessor()
{
    workerStopRequested.store(true, std::memory_order_relaxed);
    trainingJobPending.store(false, std::memory_order_relaxed);
    trainingCancelRequested.store(true, std::memory_order_relaxed);

    if (trainingThread.joinable())
        trainingThread.join();
}

juce::AudioProcessorValueTreeState::ParameterLayout MosquitobrainAudioProcessor::createParameterLayout()
{
    juce::AudioProcessorValueTreeState::ParameterLayout layout;

    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"dry_wet", 1},
        "Dry/Wet",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f),
        0.5f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int)
        { return juce::String(juce::roundToInt(value * 100.0f)) + "%"; }));

    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"train_amount", 1},
        "Train Amount",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f),
        0.7f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int)
        {
            const int passes = juce::jlimit(1, 600, 1 + juce::roundToInt(value * 599.0f));
            return juce::String(passes) + " passes";
        }));

    return layout;
}

void MosquitobrainAudioProcessor::initialiseDefaultModels()
{
    activeModel.drive = 1.2f;
    activeModel.recurrentA = 0.45f;
    activeModel.recurrentB = 0.2f;
    activeModel.bias = 0.0f;
    activeModel.outputBlend = 0.72f;
    activeModel.nnComplexity = 0.2f;
    activeModel.nnDepth = 2.0f;
    activeModel.nnResidual = 0.25f;
    activeModel.nnSaturation = 1.0f;
    activeModel.nnRecurrence = 0.35f;
    activeModel.pastContextSamples = 0.0f;
    activeModel.futureContextSamples = 0.0f;

    pendingModel = activeModel;
}

const juce::String MosquitobrainAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool MosquitobrainAudioProcessor::acceptsMidi() const { return false; }
bool MosquitobrainAudioProcessor::producesMidi() const { return false; }
bool MosquitobrainAudioProcessor::isMidiEffect() const { return false; }
double MosquitobrainAudioProcessor::getTailLengthSeconds() const { return 0.0; }

int MosquitobrainAudioProcessor::getNumPrograms() { return 1; }
int MosquitobrainAudioProcessor::getCurrentProgram() { return 0; }
void MosquitobrainAudioProcessor::setCurrentProgram(int index) { juce::ignoreUnused(index); }
const juce::String MosquitobrainAudioProcessor::getProgramName(int index)
{
    juce::ignoreUnused(index);
    return {};
}
void MosquitobrainAudioProcessor::changeProgramName(int index, const juce::String &newName)
{
    juce::ignoreUnused(index, newName);
}

void MosquitobrainAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;

    dryWetSmoothed.reset(sampleRate, 0.02);

    dryBuffer.setSize(juce::jmax(1, getTotalNumOutputChannels()), juce::jmax(1, samplesPerBlock), false, false, true);

    for (auto &channelState : inferenceState)
    {
        channelState[0] = 0.0f;
        channelState[1] = 0.0f;
    }
    for (auto &buf : contextBuffer)
        buf.fill(0.0f);
    contextWritePos.fill(0);
    contextSampleCount.fill(0);

    maxCaptureSamples = juce::jmax(1, static_cast<int>(sampleRate * 12.0));
    sourceCaptureBuffer.setSize(1, maxCaptureSamples, false, false, true);
    targetCaptureBuffer.setSize(1, maxCaptureSamples, false, false, true);
    sourceCaptureBuffer.clear();
    targetCaptureBuffer.clear();

    sourceWritePosition.store(0, std::memory_order_relaxed);
    targetWritePosition.store(0, std::memory_order_relaxed);
    captureMode.store(kCaptureNone, std::memory_order_relaxed);
}

void MosquitobrainAudioProcessor::releaseResources()
{
}

bool MosquitobrainAudioProcessor::isBusesLayoutSupported(const BusesLayout &layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono() && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;

    return true;
}

float MosquitobrainAudioProcessor::runInferenceSample(float inputSample,
                                                      float contextPast,
                                                      float contextFuture,
                                                      int channel) noexcept
{
    const auto &model = activeModel;

    auto &stateA = inferenceState[static_cast<size_t>(channel)][0];
    auto &stateB = inferenceState[static_cast<size_t>(channel)][1];

    const float memoryCoeffA = 0.02f + std::abs(model.recurrentA) * 0.12f;
    const float memoryCoeffB = 0.01f + std::abs(model.recurrentB) * 0.08f;

    stateA += (inputSample - stateA) * memoryCoeffA;
    stateB += (stateA - stateB) * memoryCoeffB;

    const float contextFeature = 0.5f * (contextPast + contextFuture);
    const float hidden1 = std::tanh(inputSample * model.drive +
                                    contextFeature +
                                    stateA * model.recurrentA +
                                    model.bias);
    const float hidden2 = std::tanh(hidden1 * (1.0f + model.recurrentB) + stateB * 0.35f - model.bias * 0.6f);

    const float filterOutput = hidden2;

    const float complexity = juce::jlimit(0.0f, 1.0f, model.nnComplexity);
    const int depth = juce::jlimit(1, 6, juce::roundToInt(model.nnDepth));
    const float residual = juce::jlimit(0.0f, 1.0f, model.nnResidual);
    const float saturation = juce::jlimit(0.2f, 2.5f, model.nnSaturation);
    const float recurrence = juce::jlimit(0.0f, 1.0f, model.nnRecurrence);

    float deep = inputSample;
    for (int layer = 0; layer < depth; ++layer)
    {
        const float layerFactor = 1.0f + 0.18f * static_cast<float>(layer);
        const float layerInput = deep * model.drive * (0.62f + 0.07f * static_cast<float>(layer)) +
                                 contextFeature +
                                 stateA * model.recurrentA * layerFactor +
                                 stateB * model.recurrentB * (1.0f - 0.05f * static_cast<float>(layer)) +
                                 model.bias;

        const float activated = std::tanh(layerInput * saturation);
        deep = (1.0f - recurrence) * activated + recurrence * deep;
    }

    const float deepOutput = std::tanh(deep + stateB * 0.3f - model.bias * 0.2f);
    const float hybrid = (1.0f - complexity) * filterOutput + complexity * (residual * inputSample + (1.0f - residual) * deepOutput);

    const float blend = juce::jlimit(0.0f, 1.0f, model.outputBlend);
    const float output = blend * hybrid + (1.0f - blend) * inputSample;

    if (!std::isfinite(output))
        return 0.0f;

    return juce::jlimit(-1.0f, 1.0f, output);
}

void MosquitobrainAudioProcessor::captureSample(float sample) noexcept
{
    const int mode = captureMode.load(std::memory_order_relaxed);
    if (mode == kCaptureNone || maxCaptureSamples <= 0)
        return;

    if (mode == kCaptureSource)
    {
        const int write = sourceWritePosition.fetch_add(1, std::memory_order_relaxed);
        if (write >= 0 && write < maxCaptureSamples)
            sourceCaptureBuffer.setSample(0, write, sample);
        else if (write >= maxCaptureSamples)
            sourceWritePosition.store(maxCaptureSamples, std::memory_order_relaxed);
        return;
    }

    const int write = targetWritePosition.fetch_add(1, std::memory_order_relaxed);
    if (write >= 0 && write < maxCaptureSamples)
        targetCaptureBuffer.setSample(0, write, sample);
    else if (write >= maxCaptureSamples)
        targetWritePosition.store(maxCaptureSamples, std::memory_order_relaxed);
}

bool MosquitobrainAudioProcessor::makeTrainingSnapshot(TrainingSnapshot &snapshot) const
{
    const int sourceCount = juce::jlimit(0, maxCaptureSamples, sourceWritePosition.load(std::memory_order_relaxed));
    const int targetCount = juce::jlimit(0, maxCaptureSamples, targetWritePosition.load(std::memory_order_relaxed));
    const int usable = juce::jmin(sourceCount, targetCount);
    if (usable <= 256)
        return false;

    snapshot.source.assign(static_cast<size_t>(usable), 0.0f);
    snapshot.target.assign(static_cast<size_t>(usable), 0.0f);

    const auto *src = sourceCaptureBuffer.getReadPointer(0);
    const auto *dst = targetCaptureBuffer.getReadPointer(0);

    std::copy(src, src + usable, snapshot.source.begin());
    std::copy(dst, dst + usable, snapshot.target.begin());

    snapshot.trainAmount = juce::jlimit(0.0f, 1.0f, latestTrainAmount.load(std::memory_order_relaxed));
    snapshot.nnComplexity = activeModel.nnComplexity;
    snapshot.nnDepth = activeModel.nnDepth;
    snapshot.nnResidual = activeModel.nnResidual;
    snapshot.nnSaturation = activeModel.nnSaturation;
    snapshot.nnRecurrence = activeModel.nnRecurrence;
    snapshot.pastContextSamples = activeModel.pastContextSamples;
    snapshot.futureContextSamples = activeModel.futureContextSamples;
    return true;
}

void MosquitobrainAudioProcessor::trainingWorkerLoop()
{
    while (!workerStopRequested.load(std::memory_order_relaxed))
    {
        if (!trainingJobPending.load(std::memory_order_relaxed))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(12));
            continue;
        }

        TrainingSnapshot snapshot;
        {
            std::lock_guard<std::mutex> lock(trainingMutex);
            if (!hasPendingSnapshot)
            {
                trainingJobPending.store(false, std::memory_order_relaxed);
                continue;
            }
            snapshot = pendingTrainingSnapshot;
            hasPendingSnapshot = false;
        }

        trainingCancelRequested.store(false, std::memory_order_relaxed);
        trainingInProgress.store(true, std::memory_order_relaxed);
        trainingProgress.store(0.0f, std::memory_order_relaxed);
        trainingSecondsRemaining.store(0.0f, std::memory_order_relaxed);
        trainingMse.store(0.0f, std::memory_order_relaxed);
        trainingAccuracy1.store(0.0f, std::memory_order_relaxed);
        trainingAccuracy5.store(0.0f, std::memory_order_relaxed);

        const bool trained = runTrainingPass(snapshot);

        trainingInProgress.store(false, std::memory_order_relaxed);
        trainingJobPending.store(false, std::memory_order_relaxed);
        trainingProgress.store(trained ? 1.0f : 0.0f, std::memory_order_relaxed);
        trainingSecondsRemaining.store(0.0f, std::memory_order_relaxed);
        lastCompletedTrainingSuccess.store(trained ? 1 : 0, std::memory_order_relaxed);
        completionPendingForUi.store(true, std::memory_order_release);

        if (trained)
            modelUpdatePending.store(true, std::memory_order_release);
    }
}

bool MosquitobrainAudioProcessor::runTrainingPass(const TrainingSnapshot &snapshot)
{
    if (snapshot.source.empty() || snapshot.target.empty())
        return false;

    auto model = activeModel;

    // Bake training-time architecture settings into the model.
    model.nnComplexity = snapshot.nnComplexity;
    model.nnDepth = snapshot.nnDepth;
    model.nnResidual = snapshot.nnResidual;
    model.nnSaturation = snapshot.nnSaturation;
    model.nnRecurrence = snapshot.nnRecurrence;
    model.pastContextSamples = static_cast<float>(juce::jlimit(0, kMaxContextSize, juce::roundToInt(snapshot.pastContextSamples)));
    model.futureContextSamples = static_cast<float>(juce::jlimit(0, kMaxContextSize, juce::roundToInt(snapshot.futureContextSamples)));

    const int pastCtxSamples = juce::jlimit(0, kMaxContextSize, juce::roundToInt(snapshot.pastContextSamples));
    const int futureCtxSamples = juce::jlimit(0, kMaxContextSize, juce::roundToInt(snapshot.futureContextSamples));
    const float complexity = juce::jlimit(0.0f, 1.0f, model.nnComplexity);
    const int depth = juce::jlimit(1, 6, juce::roundToInt(model.nnDepth));
    const float residual = juce::jlimit(0.0f, 1.0f, model.nnResidual);
    const float saturation = juce::jlimit(0.2f, 2.5f, model.nnSaturation);
    const float recurrence = juce::jlimit(0.0f, 1.0f, model.nnRecurrence);

    const int epochs = juce::jlimit(1, 600, 1 + juce::roundToInt(snapshot.trainAmount * 599.0f));
    const float learningRate = 0.0007f + snapshot.trainAmount * 0.0038f;

    // Compensate so gradient magnitude is consistent across all architectures.
    // Filter contributes (1-complexity) of the output; scale LR inversely.
    const float archLrScale = 1.0f / juce::jmax(0.05f, 1.0f - complexity);
    const float scaledLR = learningRate * archLrScale;

    const auto startedAt = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        if (workerStopRequested.load(std::memory_order_relaxed) || trainingCancelRequested.load(std::memory_order_relaxed))
            return false;

        float prevA = 0.0f;
        float prevB = 0.0f;
        double mseAccum = 0.0;
        int correct1Count = 0;
        int correct5Count = 0;
        int sampleCount = 0;

        const size_t startIndex = static_cast<size_t>(juce::jmax(1, pastCtxSamples));
        const size_t endExclusive = snapshot.source.size() - static_cast<size_t>(futureCtxSamples);
        if (endExclusive <= startIndex)
            return false;

        for (size_t i = startIndex; i < endExclusive; ++i)
        {
            const float x = snapshot.source[i];
            const float y = snapshot.target[i];
            float xPast = 0.0f;
            float xFuture = 0.0f;
            if (pastCtxSamples > 0)
            {
                for (int k = 1; k <= pastCtxSamples; ++k)
                    xPast += snapshot.source[i - static_cast<size_t>(k)];
                const float inv = 1.0f / static_cast<float>(pastCtxSamples);
                xPast *= inv;
            }
            if (futureCtxSamples > 0)
            {
                for (int k = 1; k <= futureCtxSamples; ++k)
                    xFuture += snapshot.source[i + static_cast<size_t>(k)];
                const float inv = 1.0f / static_cast<float>(futureCtxSamples);
                xFuture *= inv;
            }

            prevA += (x - prevA) * 0.08f;
            prevB += (prevA - prevB) * 0.06f;

            // 2-layer filter path (same as inference).
            const float contextFeature = 0.5f * (xPast + xFuture);
            const float z1 = x * model.drive +
                             contextFeature +
                             prevA * model.recurrentA +
                             model.bias;
            const float h1 = std::tanh(z1);
            const float z2 = h1 * (1.0f + model.recurrentB) + prevB * 0.35f - model.bias * 0.6f;
            const float h2 = std::tanh(z2);
            const float filterOutput = h2;

            // Deep path (architecture-defined; no gradient flows back through here).
            float deep = x;
            for (int layer = 0; layer < depth; ++layer)
            {
                const float lf = 1.0f + 0.18f * static_cast<float>(layer);
                const float li = deep * model.drive * (0.62f + 0.07f * static_cast<float>(layer)) +
                                 contextFeature +
                                 prevA * model.recurrentA * lf +
                                 prevB * model.recurrentB * (1.0f - 0.05f * static_cast<float>(layer)) +
                                 model.bias;
                const float act = std::tanh(li * saturation);
                deep = (1.0f - recurrence) * act + recurrence * deep;
            }
            const float deepOutput = std::tanh(deep + prevB * 0.3f - model.bias * 0.2f);

            // Full blended prediction matching the inference path.
            const float hybrid = (1.0f - complexity) * filterOutput + complexity * (residual * x + (1.0f - residual) * deepOutput);
            const float blend = juce::jlimit(0.0f, 1.0f, model.outputBlend);
            const float prediction = blend * hybrid + (1.0f - blend) * x;
            const float error = prediction - y;
            const float absError = std::abs(error);
            mseAccum += static_cast<double>(error) * static_cast<double>(error);
            ++sampleCount;
            if (absError <= 0.01f)
                ++correct1Count;
            if (absError <= 0.05f)
                ++correct5Count;

            const float dLossDpred = 2.0f * error;

            // outputBlend gradient: directly from hybrid vs. dry contribution.
            model.outputBlend -= learningRate * dLossDpred * (hybrid - x);

            // Gradient through filter path, scaled by its contribution to output.
            const float filterGradScale = blend * (1.0f - complexity);
            const float dh2dz2 = 1.0f - h2 * h2;
            const float commonGrad = dLossDpred * filterGradScale * dh2dz2;

            model.recurrentB -= scaledLR * commonGrad * h1;

            const float dh1dz1 = 1.0f - h1 * h1;
            const float h1Grad = commonGrad * (1.0f + model.recurrentB) * dh1dz1;
            model.drive -= scaledLR * h1Grad * x;
            model.recurrentA -= scaledLR * h1Grad * prevA;
            model.bias -= scaledLR * (h1Grad - commonGrad * 0.6f);

            model.drive = juce::jlimit(0.2f, 3.0f, model.drive);
            model.recurrentA = juce::jlimit(-1.5f, 1.5f, model.recurrentA);
            model.recurrentB = juce::jlimit(-1.0f, 1.0f, model.recurrentB);
            model.bias = juce::jlimit(-1.0f, 1.0f, model.bias);
            // Keep trained model contribution audible after successful training.
            model.outputBlend = juce::jlimit(0.35f, 0.98f, model.outputBlend);
        }

        const float progress = static_cast<float>(epoch + 1) / static_cast<float>(juce::jmax(1, epochs));
        trainingProgress.store(progress, std::memory_order_relaxed);
        if (sampleCount > 0)
        {
            const float mse = static_cast<float>(mseAccum / static_cast<double>(sampleCount));
            const float acc1 = static_cast<float>(correct1Count) / static_cast<float>(sampleCount);
            const float acc5 = static_cast<float>(correct5Count) / static_cast<float>(sampleCount);
            trainingMse.store(mse, std::memory_order_relaxed);
            trainingAccuracy1.store(acc1, std::memory_order_relaxed);
            trainingAccuracy5.store(acc5, std::memory_order_relaxed);
        }

        const auto now = std::chrono::steady_clock::now();
        const double elapsedSeconds = std::chrono::duration<double>(now - startedAt).count();
        const double avgPerEpoch = elapsedSeconds / static_cast<double>(epoch + 1);
        const double remaining = juce::jmax(0.0, avgPerEpoch * static_cast<double>(epochs - epoch - 1));
        trainingSecondsRemaining.store(static_cast<float>(remaining), std::memory_order_relaxed);
    }

    if (!std::isfinite(model.drive) || !std::isfinite(model.recurrentA) ||
        !std::isfinite(model.recurrentB) || !std::isfinite(model.bias) ||
        !std::isfinite(model.outputBlend))
        return false;

    pendingModel = model;
    return true;
}

void MosquitobrainAudioProcessor::getModelArchitecture(float &complexity, float &depth,
                                                       float &residual, float &saturation,
                                                       float &recurrence,
                                                       float &pastContextSamples,
                                                       float &futureContextSamples) const noexcept
{
    complexity = activeModel.nnComplexity;
    depth = activeModel.nnDepth;
    residual = activeModel.nnResidual;
    saturation = activeModel.nnSaturation;
    recurrence = activeModel.nnRecurrence;
    pastContextSamples = activeModel.pastContextSamples;
    futureContextSamples = activeModel.futureContextSamples;
}

void MosquitobrainAudioProcessor::processBlock(juce::AudioBuffer<float> &buffer, juce::MidiBuffer &midiMessages)
{
    const auto blockStartedAt = std::chrono::high_resolution_clock::now();

    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;

    const auto totalNumInputChannels = getTotalNumInputChannels();
    const auto totalNumOutputChannels = getTotalNumOutputChannels();

    for (auto channel = totalNumInputChannels; channel < totalNumOutputChannels; ++channel)
        buffer.clear(channel, 0, buffer.getNumSamples());

    if (buffer.getNumSamples() == 0)
        return;

    const float dryWetTarget = juce::jlimit(0.0f, 1.0f, parameters.getRawParameterValue("dry_wet")->load());
    const float trainAmount = juce::jlimit(0.0f, 1.0f, parameters.getRawParameterValue("train_amount")->load());
    latestTrainAmount.store(trainAmount, std::memory_order_relaxed);

    if (modelUpdatePending.exchange(false, std::memory_order_acq_rel))
        activeModel = pendingModel;

    const int pastCtxSamples = juce::jlimit(0, kMaxContextSize, juce::roundToInt(activeModel.pastContextSamples));
    const int futureCtxSamples = juce::jlimit(0, kMaxContextSize, juce::roundToInt(activeModel.futureContextSamples));
    if (getLatencySamples() != futureCtxSamples)
    {
        setLatencySamples(futureCtxSamples);
        for (auto &channelState : inferenceState)
        {
            channelState[0] = 0.0f;
            channelState[1] = 0.0f;
        }
        for (auto &buf : contextBuffer)
            buf.fill(0.0f);
        contextWritePos.fill(0);
        contextSampleCount.fill(0);
    }

    dryWetSmoothed.setTargetValue(dryWetTarget);

    if (dryBuffer.getNumChannels() < totalNumInputChannels || dryBuffer.getNumSamples() < buffer.getNumSamples())
    {
        jassertfalse;
        return;
    }

    for (int channel = 0; channel < totalNumInputChannels; ++channel)
        dryBuffer.copyFrom(channel, 0, buffer, channel, 0, buffer.getNumSamples());

    const int numSamples = buffer.getNumSamples();
    for (int sample = 0; sample < numSamples; ++sample)
    {
        const float wetMix = dryWetSmoothed.getNextValue();
        const float dryMix = 1.0f - wetMix;

        float captureTap = 0.0f;
        for (int channel = 0; channel < totalNumInputChannels; ++channel)
            captureTap += dryBuffer.getSample(channel, sample);
        captureTap /= static_cast<float>(juce::jmax(1, totalNumInputChannels));
        captureSample(captureTap);

        for (int channel = 0; channel < totalNumInputChannels; ++channel)
        {
            auto *wetData = buffer.getWritePointer(channel);
            const float dry = dryBuffer.getSample(channel, sample);

            auto &ring = contextBuffer[static_cast<size_t>(channel)];
            int &writePos = contextWritePos[static_cast<size_t>(channel)];
            int &sampleCount = contextSampleCount[static_cast<size_t>(channel)];

            ring[writePos] = dry;
            writePos = (writePos + 1) % kContextRingSize;
            sampleCount = juce::jmin(sampleCount + 1, kContextRingSize);

            const int neededSamples = pastCtxSamples + futureCtxSamples + 1;
            if (sampleCount < neededSamples)
            {
                wetData[sample] = 0.0f;
                continue;
            }

            const int latestPos = (writePos - 1 + kContextRingSize) % kContextRingSize;
            auto readOffset = [&](int offsetFromLatest) noexcept -> float
            {
                int idx = latestPos + offsetFromLatest;
                while (idx < 0)
                    idx += kContextRingSize;
                idx %= kContextRingSize;
                return ring[static_cast<size_t>(idx)];
            };

            const float xCenter = readOffset(-futureCtxSamples);
            float xPast = 0.0f;
            float xFuture = 0.0f;
            if (pastCtxSamples > 0)
            {
                for (int k = 1; k <= pastCtxSamples; ++k)
                    xPast += readOffset(-futureCtxSamples - k);
                const float inv = 1.0f / static_cast<float>(pastCtxSamples);
                xPast *= inv;
            }
            if (futureCtxSamples > 0)
            {
                for (int k = 1; k <= futureCtxSamples; ++k)
                    xFuture += readOffset(-futureCtxSamples + k);
                const float inv = 1.0f / static_cast<float>(futureCtxSamples);
                xFuture *= inv;
            }

            const float inferred = runInferenceSample(xCenter, xPast, xFuture, channel);
            wetData[sample] = dryMix * xCenter + wetMix * inferred;
        }
    }

    if (currentSampleRate > 10.0)
    {
        const auto blockFinishedAt = std::chrono::high_resolution_clock::now();
        const double blockSeconds = std::chrono::duration<double>(blockFinishedAt - blockStartedAt).count();
        const double audioSeconds = static_cast<double>(juce::jmax(1, buffer.getNumSamples())) / currentSampleRate;
        const float blockCpuPercent = static_cast<float>(juce::jlimit(0.0, 400.0, (blockSeconds / juce::jmax(1.0e-9, audioSeconds)) * 100.0));
        const float prev = realtimeCpuPercent.load(std::memory_order_relaxed);
        const float smoothed = (0.9f * prev) + (0.1f * blockCpuPercent);
        realtimeCpuPercent.store(smoothed, std::memory_order_relaxed);
    }
}

void MosquitobrainAudioProcessor::startSourceCapture() noexcept
{
    sourceWritePosition.store(0, std::memory_order_relaxed);
    captureMode.store(kCaptureSource, std::memory_order_relaxed);
}

void MosquitobrainAudioProcessor::startTargetCapture() noexcept
{
    targetWritePosition.store(0, std::memory_order_relaxed);
    captureMode.store(kCaptureTarget, std::memory_order_relaxed);
}

void MosquitobrainAudioProcessor::stopCapture() noexcept
{
    captureMode.store(kCaptureNone, std::memory_order_relaxed);
}

bool MosquitobrainAudioProcessor::queueTrainingFromCapturedAudio()
{
    stopCapture();

    TrainingSnapshot snapshot;
    if (!makeTrainingSnapshot(snapshot))
        return false;

    {
        std::lock_guard<std::mutex> lock(trainingMutex);
        pendingTrainingSnapshot = snapshot;
        lastTrainingSnapshot = snapshot;
        lastDryTrainingName = "Captured Dry";
        lastWetTrainingName = "Captured Wet";
        hasPendingSnapshot = true;
        hasLastSnapshot = true;
    }

    trainingCancelRequested.store(false, std::memory_order_relaxed);
    trainingJobPending.store(true, std::memory_order_relaxed);
    return true;
}

bool MosquitobrainAudioProcessor::retrainLastCapture()
{
    stopCapture();

    std::lock_guard<std::mutex> lock(trainingMutex);
    if (!hasLastSnapshot)
        return false;

    pendingTrainingSnapshot = lastTrainingSnapshot;
    pendingTrainingSnapshot.trainAmount = juce::jlimit(0.0f, 1.0f, latestTrainAmount.load(std::memory_order_relaxed));
    hasPendingSnapshot = true;
    trainingCancelRequested.store(false, std::memory_order_relaxed);
    trainingJobPending.store(true, std::memory_order_relaxed);
    return true;
}

bool MosquitobrainAudioProcessor::loadMonoSamplesFromWaveFile(const juce::File &file,
                                                              std::vector<float> &outSamples,
                                                              double &outSampleRate,
                                                              juce::String &errorMessage) const
{
    if (!file.existsAsFile())
    {
        errorMessage = "File does not exist: " + file.getFullPathName();
        return false;
    }

    juce::AudioFormatManager formatManager;
    formatManager.registerBasicFormats();

    std::unique_ptr<juce::AudioFormatReader> reader(formatManager.createReaderFor(file));
    if (reader == nullptr)
    {
        errorMessage = "Could not open audio file: " + file.getFileName();
        return false;
    }

    outSampleRate = reader->sampleRate;

    constexpr int64_t kMaxTrainingSamples = 2 * 60 * 48000; // 2 minutes at 48kHz equivalent
    const int64_t samplesToRead64 = juce::jlimit<int64_t>(1, kMaxTrainingSamples, reader->lengthInSamples);
    const int samplesToRead = static_cast<int>(samplesToRead64);
    const int numChannels = juce::jmax(1, static_cast<int>(reader->numChannels));

    juce::AudioBuffer<float> tempBuffer(numChannels, samplesToRead);
    tempBuffer.clear();
    if (!reader->read(&tempBuffer, 0, samplesToRead, 0, true, true))
    {
        errorMessage = "Failed to read audio data from: " + file.getFileName();
        return false;
    }

    outSamples.resize(static_cast<size_t>(samplesToRead));
    const float invChannels = 1.0f / static_cast<float>(numChannels);

    for (int sample = 0; sample < samplesToRead; ++sample)
    {
        float mono = 0.0f;
        for (int ch = 0; ch < numChannels; ++ch)
            mono += tempBuffer.getSample(ch, sample);
        outSamples[static_cast<size_t>(sample)] = mono * invChannels;
    }

    return true;
}

bool MosquitobrainAudioProcessor::queueTrainingFromWaveFiles(const juce::File &dryFile,
                                                             const juce::File &wetFile,
                                                             float trainAmount,
                                                             float nnComplexity,
                                                             float nnDepth,
                                                             float nnResidual,
                                                             float nnSaturation,
                                                             float nnRecurrence,
                                                             float pastContextSamples,
                                                             float futureContextSamples,
                                                             juce::String &errorMessage)
{
    stopCapture();

    std::vector<float> drySamples;
    std::vector<float> wetSamples;
    double dryRate = 0.0;
    double wetRate = 0.0;

    if (!loadMonoSamplesFromWaveFile(dryFile, drySamples, dryRate, errorMessage))
        return false;

    if (!loadMonoSamplesFromWaveFile(wetFile, wetSamples, wetRate, errorMessage))
        return false;

    const int drySampleCount = static_cast<int>(drySamples.size());
    const int wetBeforeResampleCount = static_cast<int>(wetSamples.size());

    // 1) Automatic sample-rate alignment: resample wet target into dry file sample-rate domain.
    wetSamples = resampleLinear(wetSamples, wetRate, dryRate);
    const int wetAfterResampleCount = static_cast<int>(wetSamples.size());

    // 2) Automatic time alignment: estimate a small lag and align heads before training.
    const int maxLag = juce::jmax(32, static_cast<int>(dryRate * 0.25));
    const int lag = estimateBestLagSamples(drySamples, wetSamples, maxLag, 65536);
    applyLagAlignment(drySamples, wetSamples, lag);

    const size_t usableSamples = juce::jmin(drySamples.size(), wetSamples.size());
    lastDrySamples.store(drySampleCount, std::memory_order_relaxed);
    lastWetSamplesBeforeResample.store(wetBeforeResampleCount, std::memory_order_relaxed);
    lastWetSamplesAfterResample.store(wetAfterResampleCount, std::memory_order_relaxed);
    lastUsableSamples.store(static_cast<int>(usableSamples), std::memory_order_relaxed);
    lastEstimatedLagSamples.store(lag, std::memory_order_relaxed);
    if (usableSamples < 256)
    {
        errorMessage = "Dry/Wet files are too short for training (need at least 256 samples).";
        return false;
    }

    TrainingSnapshot snapshot;
    snapshot.trainAmount = juce::jlimit(0.0f, 1.0f, trainAmount);
    snapshot.source.assign(drySamples.begin(), drySamples.begin() + static_cast<std::ptrdiff_t>(usableSamples));
    snapshot.target.assign(wetSamples.begin(), wetSamples.begin() + static_cast<std::ptrdiff_t>(usableSamples));
    snapshot.nnComplexity = nnComplexity;
    snapshot.nnDepth = nnDepth;
    snapshot.nnResidual = nnResidual;
    snapshot.nnSaturation = nnSaturation;
    snapshot.nnRecurrence = nnRecurrence;
    snapshot.pastContextSamples = static_cast<float>(juce::jlimit(0, kMaxContextSize, juce::roundToInt(pastContextSamples)));
    snapshot.futureContextSamples = static_cast<float>(juce::jlimit(0, kMaxContextSize, juce::roundToInt(futureContextSamples)));

    {
        std::lock_guard<std::mutex> lock(trainingMutex);
        pendingTrainingSnapshot = snapshot;
        lastTrainingSnapshot = snapshot;
        lastDryTrainingName = dryFile.getFileName();
        lastWetTrainingName = wetFile.getFileName();
        hasPendingSnapshot = true;
        hasLastSnapshot = true;
    }

    latestTrainAmount.store(snapshot.trainAmount, std::memory_order_relaxed);
    trainingCancelRequested.store(false, std::memory_order_relaxed);
    trainingJobPending.store(true, std::memory_order_relaxed);
    errorMessage.clear();
    return true;
}

void MosquitobrainAudioProcessor::cancelTraining() noexcept
{
    trainingCancelRequested.store(true, std::memory_order_relaxed);
    trainingJobPending.store(false, std::memory_order_relaxed);
}

bool MosquitobrainAudioProcessor::isTrainingInProgress() const noexcept
{
    return trainingInProgress.load(std::memory_order_relaxed);
}

float MosquitobrainAudioProcessor::getTrainingProgress() const noexcept
{
    return juce::jlimit(0.0f, 1.0f, trainingProgress.load(std::memory_order_relaxed));
}

float MosquitobrainAudioProcessor::getEstimatedTrainingSecondsRemaining() const noexcept
{
    return juce::jmax(0.0f, trainingSecondsRemaining.load(std::memory_order_relaxed));
}

void MosquitobrainAudioProcessor::getTrainingMetrics(float &outMse, float &outAccuracy1, float &outAccuracy5) const noexcept
{
    outMse = juce::jmax(0.0f, trainingMse.load(std::memory_order_relaxed));
    outAccuracy1 = juce::jlimit(0.0f, 1.0f, trainingAccuracy1.load(std::memory_order_relaxed));
    outAccuracy5 = juce::jlimit(0.0f, 1.0f, trainingAccuracy5.load(std::memory_order_relaxed));
}

void MosquitobrainAudioProcessor::getLastWaveFileLoadStats(int &outDrySamples,
                                                           int &outWetSamplesBeforeResample,
                                                           int &outWetSamplesAfterResample,
                                                           int &outUsableSamples,
                                                           int &outEstimatedLagSamples) const noexcept
{
    outDrySamples = juce::jmax(0, lastDrySamples.load(std::memory_order_relaxed));
    outWetSamplesBeforeResample = juce::jmax(0, lastWetSamplesBeforeResample.load(std::memory_order_relaxed));
    outWetSamplesAfterResample = juce::jmax(0, lastWetSamplesAfterResample.load(std::memory_order_relaxed));
    outUsableSamples = juce::jmax(0, lastUsableSamples.load(std::memory_order_relaxed));
    outEstimatedLagSamples = lastEstimatedLagSamples.load(std::memory_order_relaxed);
}

bool MosquitobrainAudioProcessor::getPersistedTrainingDataInfo(juce::String &outDryName,
                                                               juce::String &outWetName,
                                                               int &outUsableSamples) const
{
    std::lock_guard<std::mutex> lock(trainingMutex);
    if (!hasLastSnapshot)
        return false;

    outDryName = lastDryTrainingName;
    outWetName = lastWetTrainingName;
    outUsableSamples = juce::jmin(static_cast<int>(lastTrainingSnapshot.source.size()),
                                  static_cast<int>(lastTrainingSnapshot.target.size()));
    return true;
}

void MosquitobrainAudioProcessor::getRealtimeDiagnostics(float &outCpuPercent,
                                                         int &outLatencySamples,
                                                         int &outPastContextSamples,
                                                         int &outFutureContextSamples) const noexcept
{
    outCpuPercent = juce::jmax(0.0f, realtimeCpuPercent.load(std::memory_order_relaxed));
    outLatencySamples = getLatencySamples();
    outPastContextSamples = juce::jlimit(0, kMaxContextSize, juce::roundToInt(activeModel.pastContextSamples));
    outFutureContextSamples = juce::jlimit(0, kMaxContextSize, juce::roundToInt(activeModel.futureContextSamples));
}

void MosquitobrainAudioProcessor::clearPersistedTrainingData()
{
    std::lock_guard<std::mutex> lock(trainingMutex);
    pendingTrainingSnapshot = TrainingSnapshot{};
    lastTrainingSnapshot = TrainingSnapshot{};
    lastDryTrainingName.clear();
    lastWetTrainingName.clear();
    hasPendingSnapshot = false;
    hasLastSnapshot = false;
}

bool MosquitobrainAudioProcessor::consumeLastTrainingCompletion(bool &outSuccess) noexcept
{
    if (!completionPendingForUi.exchange(false, std::memory_order_acq_rel))
        return false;

    outSuccess = lastCompletedTrainingSuccess.load(std::memory_order_relaxed) != 0;
    return true;
}

bool MosquitobrainAudioProcessor::isCaptureActive() const noexcept
{
    return captureMode.load(std::memory_order_relaxed) != kCaptureNone;
}

bool MosquitobrainAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor *MosquitobrainAudioProcessor::createEditor()
{
    return new MosquitobrainAudioProcessorEditor(*this);
}

// CRC32 polynomial: 0xEDB88320
static uint32_t crc32Table[256];
static bool crc32TableInitialized = false;

static void initializeCRC32Table()
{
    if (crc32TableInitialized)
        return;

    for (uint32_t i = 0; i < 256; ++i)
    {
        uint32_t c = i;
        for (uint32_t j = 0; j < 8; ++j)
            c = (c & 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1);
        crc32Table[i] = c;
    }
    crc32TableInitialized = true;
}

static uint32_t calculateCRC32(const uint8_t *data, size_t length)
{
    initializeCRC32Table();
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; ++i)
        crc = crc32Table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFF;
}

void MosquitobrainAudioProcessor::getStateInformation(juce::MemoryBlock &destData)
{
    constexpr uint32_t VERSION = 7;

    juce::MemoryBlock payload;
    juce::MemoryOutputStream payloadStream(payload, false);

    const auto &model = activeModel;
    payloadStream.writeFloat(model.drive);
    payloadStream.writeFloat(model.recurrentA);
    payloadStream.writeFloat(model.recurrentB);
    payloadStream.writeFloat(model.bias);
    payloadStream.writeFloat(model.outputBlend);
    payloadStream.writeFloat(model.nnComplexity);
    payloadStream.writeFloat(model.nnDepth);
    payloadStream.writeFloat(model.nnResidual);
    payloadStream.writeFloat(model.nnSaturation);
    payloadStream.writeFloat(model.nnRecurrence);
    payloadStream.writeFloat(model.pastContextSamples);
    payloadStream.writeFloat(model.futureContextSamples);

    TrainingSnapshot persistedSnapshot;
    juce::String persistedDryName;
    juce::String persistedWetName;
    bool hasSnapshot = false;
    {
        std::lock_guard<std::mutex> lock(trainingMutex);
        if (hasLastSnapshot)
        {
            persistedSnapshot = lastTrainingSnapshot;
            persistedDryName = lastDryTrainingName;
            persistedWetName = lastWetTrainingName;
            hasSnapshot = true;
        }
    }

    payloadStream.writeInt(hasSnapshot ? 1 : 0);
    if (hasSnapshot)
    {
        payloadStream.writeFloat(persistedSnapshot.trainAmount);
        payloadStream.writeFloat(persistedSnapshot.nnComplexity);
        payloadStream.writeFloat(persistedSnapshot.nnDepth);
        payloadStream.writeFloat(persistedSnapshot.nnResidual);
        payloadStream.writeFloat(persistedSnapshot.nnSaturation);
        payloadStream.writeFloat(persistedSnapshot.nnRecurrence);
        payloadStream.writeFloat(persistedSnapshot.pastContextSamples);
        payloadStream.writeFloat(persistedSnapshot.futureContextSamples);
        payloadStream.writeInt(static_cast<int>(persistedSnapshot.source.size()));
        payloadStream.writeInt(static_cast<int>(persistedSnapshot.target.size()));
        payloadStream.writeString(persistedDryName);
        payloadStream.writeString(persistedWetName);

        for (float sample : persistedSnapshot.source)
            payloadStream.writeShort(floatToPcm16(sample));
        for (float sample : persistedSnapshot.target)
            payloadStream.writeShort(floatToPcm16(sample));
    }

    const uint32_t checksum = calculateCRC32(static_cast<const uint8_t *>(payload.getData()), payload.getSize());

    destData.reset();
    juce::MemoryOutputStream stateStream(destData, false);
    stateStream.writeInt(static_cast<int>(VERSION));
    stateStream.writeInt(static_cast<int>(payload.getSize()));
    stateStream.writeInt(static_cast<int>(checksum));
    stateStream.write(payload.getData(), payload.getSize());

    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    juce::MemoryBlock xmlBlock;
    copyXmlToBinary(*xml, xmlBlock);
    stateStream.write(xmlBlock.getData(), xmlBlock.getSize());
}

void MosquitobrainAudioProcessor::setStateInformation(const void *data, int sizeInBytes)
{
    constexpr int kMinHeaderBytes = 12;
    constexpr int kModelFloatCount = 12;
    const int minPayloadBytes = kModelFloatCount * static_cast<int>(sizeof(float));

    if (sizeInBytes < kMinHeaderBytes)
    {
        initialiseDefaultModels();
        clearPersistedTrainingData();
        return;
    }

    juce::MemoryInputStream stateStream(data, static_cast<size_t>(sizeInBytes), false);
    const uint32_t version = static_cast<uint32_t>(stateStream.readInt());
    const int payloadBytes = stateStream.readInt();
    const uint32_t storedChecksum = static_cast<uint32_t>(stateStream.readInt());

    bool modelValid = false;
    int xmlOffset = kMinHeaderBytes;

    if (version == 7 && payloadBytes >= minPayloadBytes && sizeInBytes >= (kMinHeaderBytes + payloadBytes))
    {
        const uint8_t *payloadData = static_cast<const uint8_t *>(data) + kMinHeaderBytes;
        xmlOffset += payloadBytes;

        const uint32_t computed = calculateCRC32(payloadData, static_cast<size_t>(payloadBytes));
        if (computed == storedChecksum)
        {
            juce::MemoryInputStream payloadStream(payloadData, static_cast<size_t>(payloadBytes), false);

            auto &model = activeModel;
            model.drive = payloadStream.readFloat();
            model.recurrentA = payloadStream.readFloat();
            model.recurrentB = payloadStream.readFloat();
            model.bias = payloadStream.readFloat();
            model.outputBlend = payloadStream.readFloat();
            model.nnComplexity = payloadStream.readFloat();
            model.nnDepth = payloadStream.readFloat();
            model.nnResidual = payloadStream.readFloat();
            model.nnSaturation = payloadStream.readFloat();
            model.nnRecurrence = payloadStream.readFloat();
            model.pastContextSamples = payloadStream.readFloat();
            model.futureContextSamples = payloadStream.readFloat();
            pendingModel = activeModel;

            const bool hasSnapshot = payloadStream.readInt() != 0;
            clearPersistedTrainingData();
            if (hasSnapshot)
            {
                TrainingSnapshot restoredSnapshot;
                restoredSnapshot.trainAmount = payloadStream.readFloat();
                restoredSnapshot.nnComplexity = payloadStream.readFloat();
                restoredSnapshot.nnDepth = payloadStream.readFloat();
                restoredSnapshot.nnResidual = payloadStream.readFloat();
                restoredSnapshot.nnSaturation = payloadStream.readFloat();
                restoredSnapshot.nnRecurrence = payloadStream.readFloat();
                restoredSnapshot.pastContextSamples = payloadStream.readFloat();
                restoredSnapshot.futureContextSamples = payloadStream.readFloat();

                const int sourceCount = payloadStream.readInt();
                const int targetCount = payloadStream.readInt();
                const juce::String restoredDryName = payloadStream.readString();
                const juce::String restoredWetName = payloadStream.readString();

                const bool countsValid = sourceCount >= 0 && sourceCount <= kMaxPersistedTrainingSamples &&
                                         targetCount >= 0 && targetCount <= kMaxPersistedTrainingSamples;
                const int requiredBytes = (sourceCount + targetCount) * static_cast<int>(sizeof(int16_t));
                const bool payloadSizedForSamples = payloadStream.getNumBytesRemaining() >= requiredBytes;

                if (countsValid && payloadSizedForSamples)
                {
                    restoredSnapshot.source.resize(static_cast<size_t>(sourceCount));
                    restoredSnapshot.target.resize(static_cast<size_t>(targetCount));

                    for (int i = 0; i < sourceCount; ++i)
                        restoredSnapshot.source[static_cast<size_t>(i)] = pcm16ToFloat(static_cast<int16_t>(payloadStream.readShort()));
                    for (int i = 0; i < targetCount; ++i)
                        restoredSnapshot.target[static_cast<size_t>(i)] = pcm16ToFloat(static_cast<int16_t>(payloadStream.readShort()));

                    std::lock_guard<std::mutex> lock(trainingMutex);
                    lastTrainingSnapshot = restoredSnapshot;
                    lastDryTrainingName = restoredDryName;
                    lastWetTrainingName = restoredWetName;
                    hasLastSnapshot = true;
                    hasPendingSnapshot = false;
                    latestTrainAmount.store(restoredSnapshot.trainAmount, std::memory_order_relaxed);
                    lastDrySamples.store(sourceCount, std::memory_order_relaxed);
                    lastWetSamplesBeforeResample.store(targetCount, std::memory_order_relaxed);
                    lastWetSamplesAfterResample.store(targetCount, std::memory_order_relaxed);
                    lastUsableSamples.store(juce::jmin(sourceCount, targetCount), std::memory_order_relaxed);
                    lastEstimatedLagSamples.store(0, std::memory_order_relaxed);
                }
            }

            modelValid = true;
        }
        else
        {
            DBG("Mosquitobrain: State corruption detected (checksum mismatch). Initializing defaults.");
        }
    }

    if (!modelValid)
    {
        initialiseDefaultModels();
        clearPersistedTrainingData();
    }

    const uint8_t *xmlData = static_cast<const uint8_t *>(data) + xmlOffset;
    const int xmlSize = sizeInBytes - xmlOffset;

    if (xmlSize <= 0)
        return;

    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(xmlData, xmlSize));
    if (xmlState.get() != nullptr && xmlState->hasTagName(parameters.state.getType()))
        parameters.replaceState(juce::ValueTree::fromXml(*xmlState));
}

juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter()
{
    return new MosquitobrainAudioProcessor();
}
