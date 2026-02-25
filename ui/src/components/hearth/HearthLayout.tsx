import { Flame, MessageSquare, BookMarked, SlidersHorizontal, ThumbsUp } from 'lucide-react';
import { ToolHeader } from '@/components/shell/ToolHeader';

export function HearthLayout() {
  return (
    <div className="flex flex-col h-full">
      <ToolHeader
        icon={Flame}
        title="Hearth"
        color="#D4A058"
      />

      <div className="flex-1 flex items-center justify-center">
        <div className="text-center max-w-md animate-fade-in">
          <div
            className="w-16 h-16 rounded-2xl mx-auto mb-6 flex items-center justify-center"
            style={{ background: 'rgba(212, 160, 88, 0.08)' }}
          >
            <Flame size={28} className="text-hearth-glow" strokeWidth={1.5} />
          </div>

          <h2 className="font-display text-xl font-semibold text-kiln-200 mb-2">
            Inference & Chat
          </h2>
          <p className="text-sm text-kiln-400 mb-8 leading-relaxed">
            Query your fine-tuned models with document context.
            Citations, model switching, and feedback capture.
          </p>

          <div className="grid grid-cols-2 gap-3 text-left">
            {[
              { icon: MessageSquare, label: 'Chat Interface', desc: 'Streaming responses' },
              { icon: BookMarked, label: 'Citations', desc: 'Source attribution' },
              { icon: SlidersHorizontal, label: 'Model Switcher', desc: 'Hot-swap LoRAs' },
              { icon: ThumbsUp, label: 'Feedback', desc: 'Route to improvement' },
            ].map(({ icon: Icon, label, desc }) => (
              <div
                key={label}
                className="card p-3 flex items-start gap-3"
              >
                <Icon size={16} className="text-hearth-glow mt-0.5 flex-shrink-0" strokeWidth={1.5} />
                <div>
                  <div className="text-xs font-medium text-kiln-200">{label}</div>
                  <div className="text-2xs text-kiln-500">{desc}</div>
                </div>
              </div>
            ))}
          </div>

          <p className="text-2xs text-kiln-600 mt-8">
            Backend API required — Sprint 13
          </p>
        </div>
      </div>
    </div>
  );
}
