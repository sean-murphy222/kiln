import { useState, useCallback } from "react";
import {
  BookOpen,
  ChevronRight,
  ChevronLeft,
  SkipForward,
  CheckCircle2,
  Circle,
  RotateCcw,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { useForgeStore } from "@/store/useForgeStore";

const DEMO_QUESTIONS = [
  {
    id: "q1",
    text: "What is the primary focus area of this discipline?",
    phase: "Scope",
  },
  {
    id: "q2",
    text: "What types of equipment or systems does this discipline cover?",
    phase: "Scope",
  },
  {
    id: "q3",
    text: "What are the most common tasks a practitioner performs daily?",
    phase: "Tasks",
  },
  {
    id: "q4",
    text: "What are the most critical safety procedures?",
    phase: "Tasks",
  },
  {
    id: "q5",
    text: "What distinguishes an expert from a novice in this discipline?",
    phase: "Expertise",
  },
  {
    id: "q6",
    text: "What are the most common mistakes or misconceptions?",
    phase: "Expertise",
  },
  {
    id: "q7",
    text: "What reference materials does a practitioner consult regularly?",
    phase: "Resources",
  },
  {
    id: "q8",
    text: "What specialized vocabulary or terminology is essential?",
    phase: "Resources",
  },
];

const PHASES = ["Scope", "Tasks", "Expertise", "Resources"];

function PhaseIndicator({
  currentPhase,
  phases,
}: {
  currentPhase: string;
  phases: string[];
}) {
  return (
    <div className="flex items-center gap-1">
      {phases.map((phase, i) => {
        const isCurrent = phase === currentPhase;
        const isPast = phases.indexOf(currentPhase) > i;
        return (
          <div key={phase} className="flex items-center gap-1">
            {i > 0 && (
              <div
                className={cn(
                  "w-8 h-px",
                  isPast ? "bg-forge-heat" : "bg-kiln-600",
                )}
              />
            )}
            <div className="flex items-center gap-1.5">
              {isPast ? (
                <CheckCircle2 size={14} className="text-forge-heat" />
              ) : (
                <Circle
                  size={14}
                  className={cn(
                    isCurrent ? "text-forge-heat" : "text-kiln-600",
                  )}
                  fill={isCurrent ? "rgba(212, 145, 92, 0.2)" : "none"}
                />
              )}
              <span
                className={cn(
                  "text-2xs font-medium",
                  isCurrent
                    ? "text-forge-heat"
                    : isPast
                      ? "text-kiln-300"
                      : "text-kiln-500",
                )}
              >
                {phase}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function DiscoveryWizard() {
  const {
    discoveryQuestions: storeQuestions,
    discoveryAnswers,
    discoveryCurrentIndex,
    setDiscoveryAnswer,
    setDiscoveryIndex,
    resetDiscovery,
    selectedDisciplineId,
  } = useForgeStore();

  // Use demo questions if store has none
  const questions = storeQuestions.length > 0 ? storeQuestions : DEMO_QUESTIONS;
  const [currentAnswer, setCurrentAnswer] = useState("");
  const [showReview, setShowReview] = useState(false);

  const currentQuestion = questions[discoveryCurrentIndex];
  const currentPhase = currentQuestion?.phase ?? PHASES[0];
  const answeredCount = discoveryAnswers.length;
  const totalCount = questions.length;
  const progressPct =
    totalCount > 0 ? Math.round((answeredCount / totalCount) * 100) : 0;

  // Get saved answer for current question
  const savedAnswer = discoveryAnswers.find(
    (a) => a.question_id === currentQuestion?.id,
  )?.answer;

  const handleNext = useCallback(() => {
    if (currentAnswer.trim() && currentQuestion) {
      setDiscoveryAnswer(currentQuestion.id, currentAnswer.trim());
    }
    if (discoveryCurrentIndex < questions.length - 1) {
      setDiscoveryIndex(discoveryCurrentIndex + 1);
      const nextQ = questions[discoveryCurrentIndex + 1];
      const nextSaved = discoveryAnswers.find(
        (a) => a.question_id === nextQ.id,
      )?.answer;
      setCurrentAnswer(nextSaved ?? "");
    } else {
      setShowReview(true);
    }
  }, [
    currentAnswer,
    currentQuestion,
    discoveryCurrentIndex,
    questions,
    discoveryAnswers,
    setDiscoveryAnswer,
    setDiscoveryIndex,
  ]);

  const handlePrev = useCallback(() => {
    if (currentAnswer.trim() && currentQuestion) {
      setDiscoveryAnswer(currentQuestion.id, currentAnswer.trim());
    }
    if (discoveryCurrentIndex > 0) {
      setDiscoveryIndex(discoveryCurrentIndex - 1);
      const prevQ = questions[discoveryCurrentIndex - 1];
      const prevSaved = discoveryAnswers.find(
        (a) => a.question_id === prevQ.id,
      )?.answer;
      setCurrentAnswer(prevSaved ?? "");
    }
  }, [
    currentAnswer,
    currentQuestion,
    discoveryCurrentIndex,
    questions,
    discoveryAnswers,
    setDiscoveryAnswer,
    setDiscoveryIndex,
  ]);

  const handleSkip = useCallback(() => {
    if (discoveryCurrentIndex < questions.length - 1) {
      setDiscoveryIndex(discoveryCurrentIndex + 1);
      const nextQ = questions[discoveryCurrentIndex + 1];
      const nextSaved = discoveryAnswers.find(
        (a) => a.question_id === nextQ.id,
      )?.answer;
      setCurrentAnswer(nextSaved ?? "");
    } else {
      setShowReview(true);
    }
  }, [discoveryCurrentIndex, questions, discoveryAnswers, setDiscoveryIndex]);

  // Initialize current answer from store on first render
  useState(() => {
    if (savedAnswer) setCurrentAnswer(savedAnswer);
  });

  if (!selectedDisciplineId) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <BookOpen
            size={32}
            className="mx-auto mb-3 text-kiln-500"
            strokeWidth={1.5}
          />
          <p className="text-sm text-kiln-400">
            Select a discipline to begin discovery
          </p>
        </div>
      </div>
    );
  }

  if (showReview) {
    return (
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="font-display text-lg font-semibold text-kiln-100 mb-1">
                Discovery Review
              </h2>
              <p className="text-sm text-kiln-400">
                {answeredCount} of {totalCount} questions answered
              </p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => {
                  setShowReview(false);
                  setDiscoveryIndex(0);
                  const firstSaved = discoveryAnswers.find(
                    (a) => a.question_id === questions[0]?.id,
                  )?.answer;
                  setCurrentAnswer(firstSaved ?? "");
                }}
                className="btn-secondary btn-sm"
              >
                <RotateCcw size={13} />
                Edit Answers
              </button>
              <button className="btn-primary btn-sm">
                <CheckCircle2 size={13} />
                Complete Discovery
              </button>
            </div>
          </div>

          <div className="space-y-3">
            {questions.map((q, i) => {
              const answer = discoveryAnswers.find(
                (a) => a.question_id === q.id,
              )?.answer;
              return (
                <div
                  key={q.id}
                  className={cn("card p-4", !answer && "border-warning/20")}
                >
                  <div className="flex items-start gap-3">
                    <span className="text-2xs text-kiln-500 tabular-nums mt-0.5 w-5 shrink-0">
                      {i + 1}.
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-2xs font-medium text-forge-heat">
                          {q.phase}
                        </span>
                      </div>
                      <p className="text-sm text-kiln-200 mb-2">{q.text}</p>
                      {answer ? (
                        <p className="text-sm text-kiln-400 leading-relaxed whitespace-pre-wrap">
                          {answer}
                        </p>
                      ) : (
                        <p className="text-sm text-warning italic">Skipped</p>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col p-6">
      <div className="max-w-2xl mx-auto w-full flex-1 flex flex-col">
        {/* Phase indicator */}
        <div className="flex items-center justify-between mb-6">
          <PhaseIndicator currentPhase={currentPhase} phases={PHASES} />
          <span className="text-2xs text-kiln-500 tabular-nums">
            {discoveryCurrentIndex + 1} / {totalCount}
          </span>
        </div>

        {/* Progress bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-1">
            <span className="text-2xs text-kiln-400">Progress</span>
            <span className="text-2xs text-kiln-500 tabular-nums">
              {progressPct}%
            </span>
          </div>
          <div className="h-1 bg-kiln-700 rounded-full">
            <div
              className="h-full bg-forge-heat rounded-full transition-all duration-300"
              style={{ width: `${progressPct}%` }}
            />
          </div>
        </div>

        {/* Question card */}
        <div
          className="card p-6 mb-6 animate-fade-in"
          key={currentQuestion?.id}
        >
          <div className="flex items-center gap-2 mb-4">
            <span className="badge-forge text-2xs">{currentPhase}</span>
            <span className="text-2xs text-kiln-500">
              Question {discoveryCurrentIndex + 1}
            </span>
          </div>

          <p className="text-base text-kiln-100 font-medium mb-6 leading-relaxed">
            {currentQuestion?.text}
          </p>

          <textarea
            value={currentAnswer}
            onChange={(e) => setCurrentAnswer(e.target.value)}
            placeholder="Type your answer here..."
            className="input-field min-h-[120px] resize-y"
            rows={4}
          />
        </div>

        {/* Navigation */}
        <div className="flex items-center justify-between">
          <button
            onClick={handlePrev}
            disabled={discoveryCurrentIndex === 0}
            className={cn(
              "btn-ghost btn-sm",
              discoveryCurrentIndex === 0 && "opacity-40 pointer-events-none",
            )}
          >
            <ChevronLeft size={14} />
            Previous
          </button>

          <button
            onClick={handleSkip}
            className="btn-ghost btn-sm text-kiln-500"
          >
            <SkipForward size={13} />
            Skip
          </button>

          <button onClick={handleNext} className="btn-primary btn-sm">
            {discoveryCurrentIndex === questions.length - 1 ? "Review" : "Next"}
            <ChevronRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
