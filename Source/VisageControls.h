#pragma once

#include <visage/ui.h>
#include <visage/graphics.h>

/**
 * PHASE 4.1.4: UI Training Control Integration
 * ============================================
 *
 * Visage UI with training control state display:
 * - Capture mode indicator (Source / Target / Idle)
 * - Training status (Idle / Training)
 * - Train Amount slider visualization
 * - Training progress display
 *
 * Note: Button interaction handled at Editor level via callbacks.
 *       This class focuses on state visualization only.
 */

class VisageMainView : public visage::Frame
{
public:
    VisageMainView()
        : captureMode(0), // 0=none, 1=source, 2=target
          isTraining(false),
          trainAmount(0.7f),
          trainingProgress(0.0f)
    {
    }

    // Update state from processor
    void setTrainingState(bool training)
    {
        if (isTraining != training)
        {
            isTraining = training;
            redraw();
        }
    }

    void setCaptureMode(int mode)
    {
        if (captureMode != mode)
        {
            captureMode = mode;
            redraw();
        }
    }

    void setTrainAmount(float amount)
    {
        if (trainAmount != amount)
        {
            trainAmount = amount;
            redraw();
        }
    }

    void setTrainingProgress(float progress)
    {
        trainingProgress = progress;
    }

    int getCaptureMode() const { return captureMode; }
    float getTrainAmount() const { return trainAmount; }

    void resized() override {}

    void draw(visage::Canvas &canvas) override
    {
        const float w = width();
        const float h = height();

        // Background
        canvas.setColor(0xff0F141A);
        canvas.fill(0, 0, w, h);

        // Main panel
        const float panelW = 620.0f;
        const float panelH = 360.0f;
        const float panelX = (w - panelW) * 0.5f;
        const float panelY = (h - panelH) * 0.5f;

        canvas.setColor(0xff1E2935);
        canvas.fill(panelX, panelY, panelW, panelH);

        // Draw header background
        canvas.setColor(0xff0F141A);
        canvas.fill(panelX, panelY, panelW, 40.0f);

        // Header border (using thin fill rectangles instead of ring)
        canvas.setColor(0xff48D0A8);
        canvas.fill(panelX, panelY, panelW, 2.0f);         // Top
        canvas.fill(panelX, panelY + 38.0f, panelW, 2.0f); // Bottom

        // Title text area (placeholder)
        canvas.fill(panelX + 10.0f, panelY + 8.0f, 150.0f, 24.0f);

        // Control section
        const float controlY = panelY + 50.0f;

        // Capture mode indicator
        canvas.setColor(0xff445555);
        canvas.fill(panelX + 20.0f, controlY, 100.0f, 40.0f);

        // Capture status light
        if (captureMode == 1)
            canvas.setColor(0xff48D0A8); // Green: source capturing
        else if (captureMode == 2)
            canvas.setColor(0xffFF8C42); // Orange: target capturing
        else
            canvas.setColor(0xff666666); // Gray: idle

        canvas.fill(panelX + 30.0f, controlY + 12.0f, 80.0f, 16.0f);

        // Training status indicator
        canvas.setColor(0xff445555);
        canvas.fill(panelX + 140.0f, controlY, 100.0f, 40.0f);

        // Training status light
        if (isTraining)
            canvas.setColor(0xffFF8C42); // Orange: training
        else if (captureMode > 0)
            canvas.setColor(0xff2E8B73); // Green: ready
        else
            canvas.setColor(0xff555555); // Gray: idle

        canvas.fill(panelX + 150.0f, controlY + 12.0f, 80.0f, 16.0f);

        // Slider area (train amount visualization)
        canvas.setColor(0xff445555);
        canvas.fill(panelX + 260.0f, controlY, 150.0f, 40.0f);

        // Slider track
        canvas.setColor(0xff263445);
        canvas.fill(panelX + 270.0f, controlY + 16.0f, 130.0f, 8.0f);

        // Slider thumb (shows train_amount)
        float thumbX = panelX + 270.0f + (trainAmount * 130.0f) - 6.0f;
        canvas.setColor(0xff2E8B73);
        canvas.fill(thumbX, controlY + 10.0f, 12.0f, 20.0f);

        // Progress indicator
        canvas.setColor(0xff445555);
        canvas.fill(panelX + 430.0f, controlY, 190.0f, 40.0f);

        // Progress bar
        if (isTraining && trainingProgress > 0.0f)
        {
            canvas.setColor(0xffFF8C42);
            canvas.fill(panelX + 440.0f, controlY + 16.0f, (trainingProgress * 170.0f), 8.0f);
        }

        // Status text area
        canvas.setColor(0xff263445);
        canvas.fill(panelX + 20.0f, controlY + 60.0f, panelW - 40.0f, 30.0f);

        // Status indicator dot
        if (isTraining)
            canvas.setColor(0xffFF8C42); // Orange: training
        else if (captureMode > 0)
            canvas.setColor(0xff2E8B73); // Green: capturing
        else
            canvas.setColor(0xff555555); // Gray: idle

        canvas.fill(panelX + 30.0f, controlY + 70.0f, 10.0f, 10.0f);

        // Model parameter visualization (3 knob wells)
        const float knobY = panelY + 180.0f;
        drawKnobWell(canvas, panelX + 80.0f, knobY, 88.0f);
        drawKnobWell(canvas, panelX + 226.0f, knobY, 88.0f);
        drawKnobWell(canvas, panelX + 372.0f, knobY, 88.0f);

        // Model slot activity indicators
        canvas.setColor(0xff2E8B73);
        canvas.fill(panelX + 490.0f, knobY - 14.0f, 20.0f, 4.0f);
        canvas.fill(panelX + 520.0f, knobY - 14.0f, 20.0f, 4.0f);
        canvas.fill(panelX + 550.0f, knobY - 14.0f, 20.0f, 4.0f);
    }

private:
    int captureMode; // 0=none, 1=source, 2=target
    bool isTraining;
    float trainAmount;
    float trainingProgress;

    void drawKnobWell(visage::Canvas &canvas, float x, float y, float size)
    {
        canvas.setColor(0xff263445);
        canvas.fill(x, y, size, size);
        canvas.setColor(0xff48D0A8);
        canvas.fill(x + 8.0f, y + 8.0f, size - 16.0f, 6.0f);
    }
};
