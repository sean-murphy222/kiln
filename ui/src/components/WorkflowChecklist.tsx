/**
 * Workflow Checklist - Shows progress through the diagnostic workflow
 */

import { CheckCircle, Circle, ArrowRight } from 'lucide-react';

interface WorkflowStep {
  id: string;
  label: string;
  description: string;
  completed: boolean;
  active: boolean;
}

interface WorkflowChecklistProps {
  hasDocument: boolean;
  hasProblems: boolean;
  hasFixPlan: boolean;
  hasAppliedFixes: boolean;
}

export function WorkflowChecklist({
  hasDocument,
  hasProblems,
  hasFixPlan,
  hasAppliedFixes,
}: WorkflowChecklistProps) {
  const steps: WorkflowStep[] = [
    {
      id: 'upload',
      label: 'Upload Document',
      description: 'Add a PDF to analyze',
      completed: hasDocument,
      active: !hasDocument,
    },
    {
      id: 'diagnose',
      label: 'Run Diagnostics',
      description: 'Detect chunk problems',
      completed: hasProblems,
      active: hasDocument && !hasProblems,
    },
    {
      id: 'preview',
      label: 'Preview Fixes',
      description: 'See automatic fixes',
      completed: hasFixPlan,
      active: hasProblems && !hasFixPlan,
    },
    {
      id: 'apply',
      label: 'Apply Fixes',
      description: 'Execute improvements',
      completed: hasAppliedFixes,
      active: hasFixPlan && !hasAppliedFixes,
    },
  ];

  const currentStepIndex = steps.findIndex(s => s.active);
  const completedCount = steps.filter(s => s.completed).length;

  return (
    <div className="bg-kiln-800 border-2 border-black p-4">
      {/* Header */}
      <div className="mb-4">
        <h3 className="text-pixel text-sm text-kiln-300">WORKFLOW GUIDE</h3>
        <p className="text-xs text-kiln-500 mt-1">
          {completedCount}/{steps.length} steps completed
        </p>
      </div>

      {/* Progress bar */}
      <div className="mb-4 bg-kiln-900 border border-black h-2 rounded-full overflow-hidden">
        <div
          className="h-full bg-ember transition-all duration-500"
          style={{ width: `${(completedCount / steps.length) * 100}%` }}
        />
      </div>

      {/* Steps */}
      <div className="space-y-2">
        {steps.map((step, index) => (
          <div
            key={step.id}
            className={`
              border-2 p-3 transition-all
              ${step.active ? 'border-ember bg-ember/10' : 'border-black bg-kiln-900'}
              ${step.completed ? 'opacity-60' : ''}
            `}
          >
            <div className="flex items-start gap-3">
              {/* Icon */}
              <div className="flex-shrink-0 mt-0.5">
                {step.completed ? (
                  <CheckCircle size={18} className="text-green-400" />
                ) : step.active ? (
                  <Circle size={18} className="text-ember animate-pulse" />
                ) : (
                  <Circle size={18} className="text-kiln-500" />
                )}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs text-kiln-500">Step {index + 1}</span>
                  {step.active && (
                    <span className="text-xs px-2 py-0.5 bg-ember text-black rounded font-bold">
                      CURRENT
                    </span>
                  )}
                </div>
                <div className="text-sm font-bold text-kiln-300 mb-1">
                  {step.label}
                </div>
                <div className="text-xs text-kiln-500">
                  {step.description}
                </div>
              </div>

              {/* Arrow for active step */}
              {step.active && (
                <div className="flex-shrink-0">
                  <ArrowRight size={18} className="text-ember animate-pulse" />
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Completion message */}
      {completedCount === steps.length && (
        <div className="mt-4 bg-green-900/30 border-2 border-green-400 p-3 rounded">
          <div className="flex items-center gap-2 mb-1">
            <CheckCircle size={16} className="text-green-400" />
            <span className="text-sm font-bold text-green-400">Workflow Complete!</span>
          </div>
          <p className="text-xs text-green-300">
            You can run diagnostics again to find and fix more problems.
          </p>
        </div>
      )}

      {/* Help text for current step */}
      {currentStepIndex >= 0 && (
        <div className="mt-4 bg-kiln-900 border border-black p-3">
          <p className="text-xs text-kiln-500">
            <strong className="text-kiln-300">Next Action: </strong>
            {currentStepIndex === 0 && 'Click "Add Doc" in the toolbar to upload a PDF'}
            {currentStepIndex === 1 && 'Click "RUN DIAGNOSTICS" button to analyze chunks'}
            {currentStepIndex === 2 && 'Click "PREVIEW AUTOMATIC FIXES" to see the fix plan'}
            {currentStepIndex === 3 && 'Click "APPLY FIXES" to execute improvements'}
          </p>
        </div>
      )}
    </div>
  );
}
