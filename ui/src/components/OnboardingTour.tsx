/**
 * Onboarding Tour - Guides new users through the diagnostic workflow
 */

import { useState } from 'react';
import { X, ArrowRight, CheckCircle } from 'lucide-react';

interface OnboardingTourProps {
  onComplete: () => void;
  onSkip: () => void;
}

const TOUR_STEPS = [
  {
    title: 'Welcome to CHONK Diagnostics!',
    description: 'CHONK helps you find and fix problems in your document chunks before embedding them for RAG.',
    icon: '🎯',
  },
  {
    title: 'Step 1: Upload a Document',
    description: 'Click the "Add Doc" button in the toolbar to upload a PDF. CHONK will extract text and create initial chunks.',
    highlight: 'upload',
    icon: '📄',
  },
  {
    title: 'Step 2: Run Diagnostics',
    description: 'Click "RUN DIAGNOSTICS" to analyze your chunks. CHONK will detect problems like incomplete sentences, broken references, and mixed topics.',
    highlight: 'diagnostics',
    icon: '🔍',
  },
  {
    title: 'Step 3: Review Problems',
    description: 'Browse the detected problems in the left panel. Click any problem to see the affected chunk and understand what\'s wrong.',
    highlight: 'problems',
    icon: '⚠️',
  },
  {
    title: 'Step 4: Preview Fixes',
    description: 'Click "PREVIEW AUTOMATIC FIXES" to see what CHONK can fix automatically. You\'ll see merge/split actions with confidence scores.',
    highlight: 'preview',
    icon: '🔧',
  },
  {
    title: 'Step 5: Apply Fixes',
    description: 'Click "APPLY FIXES" to execute the improvements. CHONK will merge incomplete chunks, split mixed topics, and fix structural issues.',
    highlight: 'apply',
    icon: '✨',
  },
  {
    title: 'Step 6: Measure Improvement',
    description: 'See before/after metrics showing how many problems were fixed and the improvement percentage. You can run diagnostics again for more improvements!',
    highlight: 'metrics',
    icon: '📊',
  },
  {
    title: 'Ready to Start!',
    description: 'That\'s the complete workflow! Upload a document to begin, and CHONK will guide you through each step.',
    icon: '🚀',
  },
];

export function OnboardingTour({ onComplete, onSkip }: OnboardingTourProps) {
  const [currentStep, setCurrentStep] = useState(0);

  const handleNext = () => {
    if (currentStep === TOUR_STEPS.length - 1) {
      onComplete();
    } else {
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const step = TOUR_STEPS[currentStep];
  const isLastStep = currentStep === TOUR_STEPS.length - 1;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4">
      <div className="bg-kiln-800 border-4 border-ember rounded-lg shadow-2xl max-w-2xl w-full">
        {/* Header */}
        <div className="border-b-2 border-black p-4 flex items-center justify-between bg-gradient-to-r from-ember/20 to-transparent">
          <div className="flex items-center gap-3">
            <span className="text-3xl">{step.icon}</span>
            <h2 className="text-pixel text-xl text-kiln-300">{step.title}</h2>
          </div>
          <button
            onClick={onSkip}
            className="text-kiln-500 hover:text-kiln-300 transition-colors"
            title="Skip tour"
          >
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="p-8">
          <p className="text-lg text-kiln-300 leading-relaxed mb-6">
            {step.description}
          </p>

          {/* Progress dots */}
          <div className="flex items-center justify-center gap-2 mb-6">
            {TOUR_STEPS.map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentStep(index)}
                className={`w-3 h-3 rounded-full transition-all ${
                  index === currentStep
                    ? 'bg-ember w-8'
                    : index < currentStep
                    ? 'bg-green-400'
                    : 'bg-kiln-500'
                }`}
                title={`Step ${index + 1}`}
              />
            ))}
          </div>

          {/* Step counter */}
          <div className="text-center text-sm text-kiln-500 mb-6">
            Step {currentStep + 1} of {TOUR_STEPS.length}
          </div>
        </div>

        {/* Footer */}
        <div className="border-t-2 border-black p-4 flex items-center justify-between bg-kiln-900">
          <button
            onClick={handlePrevious}
            disabled={currentStep === 0}
            className={`btn btn-secondary ${
              currentStep === 0 ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            PREVIOUS
          </button>

          <div className="flex items-center gap-3">
            <button
              onClick={onSkip}
              className="text-sm text-kiln-500 hover:text-kiln-300 transition-colors"
            >
              Skip Tour
            </button>
            <button
              onClick={handleNext}
              className="btn btn-primary flex items-center gap-2"
            >
              {isLastStep ? (
                <>
                  <CheckCircle size={16} />
                  <span>GET STARTED</span>
                </>
              ) : (
                <>
                  <span>NEXT</span>
                  <ArrowRight size={16} />
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
