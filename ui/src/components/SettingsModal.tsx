import { useState, useEffect } from 'react';
import { X, Settings, Sliders, Cpu, Palette, Save, FileSearch } from 'lucide-react';
import { useStore } from '../store/useStore';
import { settingsAPI, utilAPI, Extractor } from '../api/chonk';

interface SettingsModalProps {
  onClose: () => void;
}

interface AppSettings {
  // Chunking defaults
  default_chunker: string;
  default_target_tokens: number;
  default_max_tokens: number;
  default_min_tokens: number;
  default_overlap_tokens: number;

  // Embedding settings
  embedding_model: string;
  embedding_batch_size: number;

  // Extraction settings
  extraction_tier: string;
  extraction_auto_upgrade: boolean;

  // UI preferences
  theme: string;
  show_quality_warnings: boolean;
  auto_save: boolean;
}

const DEFAULT_SETTINGS: AppSettings = {
  default_chunker: 'hierarchy',
  default_target_tokens: 400,
  default_max_tokens: 600,
  default_min_tokens: 100,
  default_overlap_tokens: 50,
  embedding_model: 'all-MiniLM-L6-v2',
  embedding_batch_size: 32,
  extraction_tier: 'fast',
  extraction_auto_upgrade: false,
  theme: 'dark',
  show_quality_warnings: true,
  auto_save: true,
};

export function SettingsModal({ onClose }: SettingsModalProps) {
  const { setError } = useStore();
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [activeTab, setActiveTab] = useState<'extraction' | 'chunking' | 'embedding' | 'appearance'>('extraction');
  const [isSaving, setIsSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [extractors, setExtractors] = useState<Extractor[]>([]);

  // Load settings and extractors on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        const [savedSettings, extractorData] = await Promise.all([
          settingsAPI.get().catch(() => null),
          utilAPI.getExtractors().catch(() => ({ extractors: [] })),
        ]);
        if (savedSettings) {
          setSettings({ ...DEFAULT_SETTINGS, ...savedSettings } as AppSettings);
        }
        setExtractors(extractorData.extractors);
      } catch (err) {
        // Settings might not exist yet, use defaults
      }
    };
    loadData();
  }, []);

  const updateSetting = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await settingsAPI.save(settings);
      setHasChanges(false);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save settings');
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = () => {
    setSettings(DEFAULT_SETTINGS);
    setHasChanges(true);
  };

  const tabs = [
    { id: 'extraction', label: 'Extraction', icon: FileSearch },
    { id: 'chunking', label: 'Chunking', icon: Sliders },
    { id: 'embedding', label: 'Embedding', icon: Cpu },
    { id: 'appearance', label: 'Appearance', icon: Palette },
  ] as const;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="card w-full max-w-2xl max-h-[80vh] flex flex-col animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-kiln-600">
          <div className="flex items-center gap-2">
            <Settings size={18} className="text-ember" />
            <h2 className="text-sm font-medium text-kiln-100">Settings</h2>
          </div>
          <button
            className="p-1 text-kiln-500 hover:text-kiln-100"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-kiln-600">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                className={`
                  flex items-center gap-2 px-4 py-2 text-xs font-medium transition-colors
                  ${activeTab === tab.id
                    ? 'text-ember border-b-2 border-ember'
                    : 'text-kiln-500 hover:text-kiln-300'
                  }
                `}
                onClick={() => setActiveTab(tab.id)}
              >
                <Icon size={14} />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {activeTab === 'extraction' && (
            <ExtractionSettings
              settings={settings}
              updateSetting={updateSetting}
              extractors={extractors}
            />
          )}
          {activeTab === 'chunking' && (
            <ChunkingSettings settings={settings} updateSetting={updateSetting} />
          )}
          {activeTab === 'embedding' && (
            <EmbeddingSettings settings={settings} updateSetting={updateSetting} />
          )}
          {activeTab === 'appearance' && (
            <AppearanceSettings settings={settings} updateSetting={updateSetting} />
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-kiln-600">
          <button
            className="text-xs text-kiln-500 hover:text-error"
            onClick={handleReset}
          >
            Reset to Defaults
          </button>
          <div className="flex items-center gap-3">
            <button className="btn-secondary" onClick={onClose} disabled={isSaving}>
              Cancel
            </button>
            <button
              className="btn-primary flex items-center gap-2"
              onClick={handleSave}
              disabled={isSaving || !hasChanges}
            >
              <Save size={14} />
              {isSaving ? 'Saving...' : 'Save Settings'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

interface SettingsSectionProps {
  settings: AppSettings;
  updateSetting: <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => void;
}

interface ExtractionSettingsProps extends SettingsSectionProps {
  extractors: Extractor[];
}

function ExtractionSettings({
  settings,
  updateSetting,
  extractors,
}: ExtractionSettingsProps) {
  // Add auto option to extractors list
  const tierOptions = [
    {
      id: 'auto',
      name: 'Auto-Select',
      description: 'Automatically choose based on document complexity',
      available: true,
      tier: 0,
      install_hint: null,
    },
    ...extractors,
  ];

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-medium text-kiln-100 mb-4">
          PDF Extraction Engine
        </h3>
        <p className="text-xs text-kiln-500 mb-4">
          Choose how PDF documents are parsed. Higher tiers provide better
          accuracy for complex documents but may be slower.
        </p>
        <div className="space-y-2">
          {tierOptions.map((e) => (
            <label
              key={e.id}
              className={`
                flex items-start gap-3 p-3 rounded transition-colors
                ${!e.available ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                ${settings.extraction_tier === e.id
                  ? 'bg-ember/20 border border-ember'
                  : 'bg-kiln-700 border border-transparent hover:border-kiln-600'
                }
              `}
            >
              <input
                type="radio"
                name="extractor"
                value={e.id}
                checked={settings.extraction_tier === e.id}
                onChange={(ev) => updateSetting('extraction_tier', ev.target.value)}
                disabled={!e.available}
                className="mt-1"
              />
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-kiln-100">
                      {e.name}
                    </span>
                    {e.tier > 0 && (
                      <span className="px-1.5 py-0.5 text-[10px] rounded bg-kiln-600 text-kiln-300">
                        Tier {e.tier}
                      </span>
                    )}
                  </div>
                  {!e.available && e.install_hint && (
                    <code className="text-[10px] text-forge-heat bg-kiln-900 px-1.5 py-0.5 rounded">
                      {e.install_hint}
                    </code>
                  )}
                </div>
                <div className="text-xs text-kiln-500 mt-1">{e.description}</div>
              </div>
            </label>
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-medium text-kiln-100 mb-4">
          Advanced Options
        </h3>
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={settings.extraction_auto_upgrade}
            onChange={(e) => updateSetting('extraction_auto_upgrade', e.target.checked)}
            className="w-4 h-4 rounded border-kiln-600 bg-kiln-900"
          />
          <div>
            <span className="text-sm text-kiln-300">Auto-upgrade for complex documents</span>
            <p className="text-xs text-kiln-500">
              Automatically use a higher tier when many tables, scanned pages,
              or multi-column layouts are detected.
            </p>
          </div>
        </label>
      </div>

      <div className="p-3 rounded bg-kiln-700 border border-kiln-600">
        <p className="text-xs text-kiln-500">
          <span className="text-ember">Tip:</span> For most documents,
          the Fast tier provides excellent results. Use Enhanced or AI tiers
          for academic papers, scanned documents, or complex table-heavy PDFs.
        </p>
      </div>
    </div>
  );
}

function ChunkingSettings({ settings, updateSetting }: SettingsSectionProps) {
  const chunkers = [
    {
      id: 'hierarchy',
      name: 'Hierarchy',
      description: 'Respects headings and document structure (recommended)',
    },
    {
      id: 'recursive',
      name: 'Recursive',
      description: 'Splits on natural boundaries (paragraphs, sentences)',
    },
    {
      id: 'fixed',
      name: 'Fixed Size',
      description: 'Simple fixed-size chunks with overlap',
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-medium text-kiln-100 mb-4">
          Default Chunking Strategy
        </h3>
        <div className="space-y-2">
          {chunkers.map((c) => (
            <label
              key={c.id}
              className={`
                flex items-start gap-3 p-3 rounded cursor-pointer transition-colors
                ${settings.default_chunker === c.id
                  ? 'bg-ember/20 border border-ember'
                  : 'bg-kiln-700 border border-transparent hover:border-kiln-600'
                }
              `}
            >
              <input
                type="radio"
                name="chunker"
                value={c.id}
                checked={settings.default_chunker === c.id}
                onChange={(e) => updateSetting('default_chunker', e.target.value)}
                className="mt-1"
              />
              <div>
                <div className="text-sm font-medium text-kiln-100">{c.name}</div>
                <div className="text-xs text-kiln-500">{c.description}</div>
              </div>
            </label>
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-medium text-kiln-100 mb-4">
          Default Token Settings
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-kiln-500 mb-2">
              Target Tokens
            </label>
            <input
              type="number"
              className="input-field"
              value={settings.default_target_tokens}
              onChange={(e) =>
                updateSetting('default_target_tokens', parseInt(e.target.value) || 400)
              }
              min={50}
              max={2000}
            />
            <p className="text-[10px] text-kiln-600 mt-1">
              Ideal chunk size (200-500 recommended)
            </p>
          </div>
          <div>
            <label className="block text-xs text-kiln-500 mb-2">
              Max Tokens
            </label>
            <input
              type="number"
              className="input-field"
              value={settings.default_max_tokens}
              onChange={(e) =>
                updateSetting('default_max_tokens', parseInt(e.target.value) || 600)
              }
              min={100}
              max={4000}
            />
            <p className="text-[10px] text-kiln-600 mt-1">Maximum chunk size</p>
          </div>
          <div>
            <label className="block text-xs text-kiln-500 mb-2">
              Min Tokens
            </label>
            <input
              type="number"
              className="input-field"
              value={settings.default_min_tokens}
              onChange={(e) =>
                updateSetting('default_min_tokens', parseInt(e.target.value) || 100)
              }
              min={10}
              max={500}
            />
            <p className="text-[10px] text-kiln-600 mt-1">
              Merge chunks smaller than this
            </p>
          </div>
          <div>
            <label className="block text-xs text-kiln-500 mb-2">
              Overlap Tokens
            </label>
            <input
              type="number"
              className="input-field"
              value={settings.default_overlap_tokens}
              onChange={(e) =>
                updateSetting('default_overlap_tokens', parseInt(e.target.value) || 50)
              }
              min={0}
              max={200}
            />
            <p className="text-[10px] text-kiln-600 mt-1">
              Content shared between chunks
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function EmbeddingSettings({ settings, updateSetting }: SettingsSectionProps) {
  const models = [
    {
      id: 'all-MiniLM-L6-v2',
      name: 'MiniLM-L6 (Default)',
      description: 'Fast, efficient, 384 dimensions. Best for most use cases.',
      size: '~90MB',
    },
    {
      id: 'all-mpnet-base-v2',
      name: 'MPNet Base',
      description: 'Higher quality, 768 dimensions. Slower but more accurate.',
      size: '~420MB',
    },
    {
      id: 'paraphrase-MiniLM-L3-v2',
      name: 'MiniLM-L3 (Lightweight)',
      description: 'Fastest option, 384 dimensions. Good for large documents.',
      size: '~60MB',
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-medium text-kiln-100 mb-4">
          Embedding Model
        </h3>
        <div className="space-y-2">
          {models.map((m) => (
            <label
              key={m.id}
              className={`
                flex items-start gap-3 p-3 rounded cursor-pointer transition-colors
                ${settings.embedding_model === m.id
                  ? 'bg-ember/20 border border-ember'
                  : 'bg-kiln-700 border border-transparent hover:border-kiln-600'
                }
              `}
            >
              <input
                type="radio"
                name="model"
                value={m.id}
                checked={settings.embedding_model === m.id}
                onChange={(e) => updateSetting('embedding_model', e.target.value)}
                className="mt-1"
              />
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-kiln-100">
                    {m.name}
                  </span>
                  <span className="text-[10px] text-kiln-600">{m.size}</span>
                </div>
                <div className="text-xs text-kiln-500">{m.description}</div>
              </div>
            </label>
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-medium text-kiln-100 mb-4">
          Performance
        </h3>
        <div>
          <label className="block text-xs text-kiln-500 mb-2">
            Batch Size
          </label>
          <input
            type="number"
            className="input-field w-32"
            value={settings.embedding_batch_size}
            onChange={(e) =>
              updateSetting('embedding_batch_size', parseInt(e.target.value) || 32)
            }
            min={1}
            max={128}
          />
          <p className="text-[10px] text-kiln-600 mt-1">
            Number of chunks to embed at once. Higher = faster but uses more memory.
          </p>
        </div>
      </div>

      <div className="p-3 rounded bg-kiln-700 border border-kiln-600">
        <p className="text-xs text-kiln-500">
          <span className="text-ember">Note:</span> Changing the embedding
          model will require re-embedding all chunks. This may take a while for
          large projects.
        </p>
      </div>
    </div>
  );
}

function AppearanceSettings({ settings, updateSetting }: SettingsSectionProps) {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-sm font-medium text-kiln-100 mb-4">
          Theme
        </h3>
        <div className="flex gap-3">
          <label
            className={`
              flex-1 p-4 rounded cursor-pointer text-center transition-colors
              ${settings.theme === 'dark'
                ? 'bg-ember/20 border-2 border-ember'
                : 'bg-kiln-700 border-2 border-transparent hover:border-kiln-600'
              }
            `}
          >
            <input
              type="radio"
              name="theme"
              value="dark"
              checked={settings.theme === 'dark'}
              onChange={(e) => updateSetting('theme', e.target.value)}
              className="sr-only"
            />
            <div className="w-12 h-8 mx-auto mb-2 rounded bg-kiln-950 border border-kiln-600" />
            <span className="text-sm text-kiln-300">Dark</span>
          </label>
          <label
            className={`
              flex-1 p-4 rounded cursor-pointer text-center transition-colors opacity-50
              ${settings.theme === 'light'
                ? 'bg-ember/20 border-2 border-ember'
                : 'bg-kiln-700 border-2 border-transparent'
              }
            `}
            title="Coming soon"
          >
            <input
              type="radio"
              name="theme"
              value="light"
              disabled
              className="sr-only"
            />
            <div className="w-12 h-8 mx-auto mb-2 rounded bg-gray-200 border border-gray-300" />
            <span className="text-sm text-kiln-500">Light (Soon)</span>
          </label>
        </div>
      </div>

      <div>
        <h3 className="text-sm font-medium text-kiln-100 mb-4">
          Quality Indicators
        </h3>
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={settings.show_quality_warnings}
            onChange={(e) => updateSetting('show_quality_warnings', e.target.checked)}
            className="w-4 h-4 rounded border-kiln-600 bg-kiln-900"
          />
          <div>
            <span className="text-sm text-kiln-300">Show quality warnings</span>
            <p className="text-xs text-kiln-500">
              Display warnings for chunks with low quality scores
            </p>
          </div>
        </label>
      </div>

      <div>
        <h3 className="text-sm font-medium text-kiln-100 mb-4">
          Auto-Save
        </h3>
        <label className="flex items-center gap-3 cursor-pointer">
          <input
            type="checkbox"
            checked={settings.auto_save}
            onChange={(e) => updateSetting('auto_save', e.target.checked)}
            className="w-4 h-4 rounded border-kiln-600 bg-kiln-900"
          />
          <div>
            <span className="text-sm text-kiln-300">
              Automatically save project changes
            </span>
            <p className="text-xs text-kiln-500">
              Save changes automatically when editing chunks or metadata
            </p>
          </div>
        </label>
      </div>
    </div>
  );
}
