/**
 * Diagnostic Welcome Screen - Shows capabilities when no document loaded
 */

import { Search, Zap, BarChart, CheckCircle, AlertTriangle } from 'lucide-react';

export function DiagnosticWelcome() {
  return (
    <div className="h-full overflow-y-auto bg-surface-bg">
      <div className="flex items-center justify-center p-8">
        <div className="max-w-4xl">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-pixel text-4xl text-chonk-light mb-4">
            CHONK Diagnostics
          </h1>
          <p className="text-lg text-chonk-gray mb-6">
            Find and fix chunk problems <strong className="text-chonk-light">before</strong> embedding
          </p>
          <div className="inline-block bg-accent-primary/20 border-2 border-accent-primary px-4 py-2 rounded">
            <p className="text-sm text-accent-primary font-bold">
              ✨ 100% Heuristic-Based • No API Key Required • Zero LLM Costs
            </p>
          </div>
        </div>

        {/* What It Does */}
        <div className="bg-surface-panel border-4 border-black p-8 mb-8">
          <h2 className="text-pixel text-xl text-chonk-light mb-6 flex items-center gap-2">
            <Search size={24} className="text-accent-primary" />
            What CHONK Detects
          </h2>

          <div className="grid grid-cols-2 gap-6">
            <div className="bg-surface-bg border-2 border-black p-4">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle size={18} className="text-red-400" />
                <h3 className="text-sm font-bold text-red-400">Semantic Incompleteness</h3>
              </div>
              <p className="text-xs text-chonk-gray">
                Partial ideas, dangling connectives ("however", "therefore"), incomplete sentences
              </p>
            </div>

            <div className="bg-surface-bg border-2 border-black p-4">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle size={18} className="text-orange-400" />
                <h3 className="text-sm font-bold text-orange-400">Semantic Contamination</h3>
              </div>
              <p className="text-xs text-chonk-gray">
                Multiple unrelated topics mixed together in one chunk
              </p>
            </div>

            <div className="bg-surface-bg border-2 border-black p-4">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle size={18} className="text-yellow-400" />
                <h3 className="text-sm font-bold text-yellow-400">Structural Breakage</h3>
              </div>
              <p className="text-xs text-chonk-gray">
                Split lists, broken tables, incomplete procedures
              </p>
            </div>

            <div className="bg-surface-bg border-2 border-black p-4">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle size={18} className="text-blue-400" />
                <h3 className="text-sm font-bold text-blue-400">Reference Orphaning</h3>
              </div>
              <p className="text-xs text-chonk-gray">
                Broken references ("as follows", "see above", "mentioned earlier")
              </p>
            </div>
          </div>
        </div>

        {/* Workflow */}
        <div className="bg-surface-panel border-4 border-black p-8 mb-8">
          <h2 className="text-pixel text-xl text-chonk-light mb-6 flex items-center gap-2">
            <Zap size={24} className="text-accent-primary" />
            How It Works
          </h2>

          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-accent-primary text-black rounded-full flex items-center justify-center font-bold">
                1
              </div>
              <div>
                <h3 className="text-sm font-bold text-chonk-light mb-1">Upload PDF Document</h3>
                <p className="text-xs text-chonk-gray">
                  CHONK extracts text and creates initial chunks using Docling (best-in-class structure detection)
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-accent-primary text-black rounded-full flex items-center justify-center font-bold">
                2
              </div>
              <div>
                <h3 className="text-sm font-bold text-chonk-light mb-1">Run Diagnostics</h3>
                <p className="text-xs text-chonk-gray">
                  Analyze chunks with 10+ heuristic checks. Generate test questions to validate retrieval quality.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-accent-primary text-black rounded-full flex items-center justify-center font-bold">
                3
              </div>
              <div>
                <h3 className="text-sm font-bold text-chonk-light mb-1">Preview Automatic Fixes</h3>
                <p className="text-xs text-chonk-gray">
                  See merge/split actions with confidence scores. Review before applying.
                </p>
              </div>
            </div>

            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-accent-primary text-black rounded-full flex items-center justify-center font-bold">
                4
              </div>
              <div>
                <h3 className="text-sm font-bold text-chonk-light mb-1">Apply Fixes & Measure</h3>
                <p className="text-xs text-chonk-gray">
                  Execute improvements. See before/after metrics showing problems fixed and improvement %.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Example Results */}
        <div className="bg-green-900/20 border-2 border-green-400 p-6 mb-8">
          <h2 className="text-pixel text-lg text-green-400 mb-4 flex items-center gap-2">
            <BarChart size={20} />
            Example Results (MIL-STD Technical Document)
          </h2>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-chonk-gray">Problems Detected</div>
              <div className="text-2xl font-bold text-green-400">89</div>
              <div className="text-xs text-green-300">in 50 chunks</div>
            </div>
            <div>
              <div className="text-chonk-gray">Automatic Fixes</div>
              <div className="text-2xl font-bold text-green-400">5</div>
              <div className="text-xs text-green-300">merge operations</div>
            </div>
            <div>
              <div className="text-chonk-gray">Improvement</div>
              <div className="text-2xl font-bold text-green-400">6.7%</div>
              <div className="text-xs text-green-300">first pass</div>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="text-center">
          <p className="text-chonk-gray mb-4">
            Click <strong className="text-accent-primary">"Add Doc"</strong> in the toolbar to upload a PDF and start analyzing
          </p>
          <div className="flex items-center justify-center gap-2 text-xs text-chonk-gray">
            <CheckCircle size={14} className="text-green-400" />
            <span>No API keys needed</span>
            <span>•</span>
            <CheckCircle size={14} className="text-green-400" />
            <span>All processing local</span>
            <span>•</span>
            <CheckCircle size={14} className="text-green-400" />
            <span>Zero cost per analysis</span>
          </div>
        </div>
        </div>
      </div>
    </div>
  );
}
