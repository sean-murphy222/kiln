import { useState, useRef, useEffect } from 'react';
import {
  Settings,
  Mountain,
  Hammer,
  Factory,
  Flame,
  Info,
  Server,
  FileBox,
  Cpu,
  MessageSquare,
  Moon,
  RotateCcw,
  Save,
  ExternalLink,
  BookOpen,
  Target,
  ToggleLeft,
  ToggleRight,
  Hash,
  Layers,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/cn';
import { ToolHeader } from '@/components/shell/ToolHeader';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface KilnSettings {
  // General
  backendUrl: string;
  theme: 'dark' | 'light';

  // Quarry
  quarryDefaultStrategy: string;
  quarryMaxFileSizeMb: number;

  // Forge
  forgeTargetExamples: number;
  forgeAutoSave: boolean;

  // Foundry
  foundryBaseModel: string;
  foundryLoraRank: number;

  // Hearth
  hearthDefaultSlot: string;
  hearthStreaming: boolean;
}

const DEFAULT_SETTINGS: KilnSettings = {
  backendUrl: 'http://127.0.0.1:8420',
  theme: 'dark',
  quarryDefaultStrategy: 'hierarchy',
  quarryMaxFileSizeMb: 100,
  forgeTargetExamples: 50,
  forgeAutoSave: true,
  foundryBaseModel: 'meta-llama/Llama-3.1-8B',
  foundryLoraRank: 16,
  hearthDefaultSlot: 'slot-0',
  hearthStreaming: true,
};

type SectionId = 'general' | 'quarry' | 'forge' | 'foundry' | 'hearth' | 'about';

interface NavItem {
  id: SectionId;
  label: string;
  icon: React.ElementType;
  color: string;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'general', label: 'General', icon: Settings, color: '#8892A8' },
  { id: 'quarry', label: 'Quarry', icon: Mountain, color: '#7C92A8' },
  { id: 'forge', label: 'Forge', icon: Hammer, color: '#D4915C' },
  { id: 'foundry', label: 'Foundry', icon: Factory, color: '#6BA089' },
  { id: 'hearth', label: 'Hearth', icon: Flame, color: '#D4A058' },
  { id: 'about', label: 'About', icon: Info, color: '#8892A8' },
];

/* ------------------------------------------------------------------ */
/*  Toggle Switch                                                      */
/* ------------------------------------------------------------------ */

interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  description?: string;
  accentColor?: string;
}

function ToggleSwitch({ checked, onChange, label, description, accentColor = '#E8734A' }: ToggleSwitchProps) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="w-full flex items-center justify-between gap-4 group"
    >
      <div className="text-left">
        <span className="text-sm text-kiln-200">{label}</span>
        {description && (
          <p className="text-2xs text-kiln-500 mt-0.5">{description}</p>
        )}
      </div>
      <div className="flex-shrink-0">
        {checked ? (
          <ToggleRight
            size={28}
            style={{ color: accentColor }}
            className="transition-colors duration-150"
          />
        ) : (
          <ToggleLeft
            size={28}
            className="text-kiln-500 group-hover:text-kiln-400 transition-colors duration-150"
          />
        )}
      </div>
    </button>
  );
}

/* ------------------------------------------------------------------ */
/*  Setting Field                                                      */
/* ------------------------------------------------------------------ */

interface SettingFieldProps {
  label: string;
  description?: string;
  children: React.ReactNode;
}

function SettingField({ label, description, children }: SettingFieldProps) {
  return (
    <div>
      <label className="block text-xs font-medium text-kiln-400 mb-1.5">
        {label}
      </label>
      {children}
      {description && (
        <p className="text-2xs text-kiln-500 mt-1">{description}</p>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Section Card                                                       */
/* ------------------------------------------------------------------ */

interface SectionCardProps {
  id: string;
  title: string;
  icon: React.ElementType;
  iconColor: string;
  children: React.ReactNode;
}

function SectionCard({ id, title, icon: Icon, iconColor, children }: SectionCardProps) {
  return (
    <section id={`section-${id}`} className="card p-5 animate-fade-in">
      <div className="flex items-center gap-3 mb-5">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{ background: `${iconColor}12` }}
        >
          <Icon size={16} style={{ color: iconColor }} strokeWidth={2} />
        </div>
        <h3 className="font-display text-sm font-semibold text-kiln-200">
          {title}
        </h3>
      </div>
      <div className="space-y-5 pl-11">
        {children}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/*  Section Components                                                 */
/* ------------------------------------------------------------------ */

interface SectionProps {
  settings: KilnSettings;
  update: <K extends keyof KilnSettings>(key: K, value: KilnSettings[K]) => void;
}

function GeneralSection({ settings, update }: SectionProps) {
  return (
    <SectionCard id="general" title="General" icon={Settings} iconColor="#8892A8">
      <SettingField
        label="Backend URL"
        description="The address where the Kiln backend API is running."
      >
        <div className="flex items-center gap-2">
          <Server size={14} className="text-kiln-500 flex-shrink-0" />
          <input
            type="url"
            value={settings.backendUrl}
            onChange={(e) => update('backendUrl', e.target.value)}
            placeholder="http://127.0.0.1:8420"
            className="input-field"
          />
        </div>
      </SettingField>

      <SettingField label="Theme">
        <div className="flex gap-3">
          <label
            className={cn(
              'flex-1 p-3 rounded-kiln cursor-pointer text-center transition-all duration-150',
              settings.theme === 'dark'
                ? 'bg-ember-faint border border-ember/40'
                : 'bg-kiln-900 border border-kiln-600 hover:border-kiln-500',
            )}
          >
            <input
              type="radio"
              name="theme"
              value="dark"
              checked={settings.theme === 'dark'}
              onChange={() => update('theme', 'dark')}
              className="sr-only"
            />
            <Moon size={18} className="mx-auto mb-2 text-kiln-300" />
            <span className="text-xs font-medium text-kiln-200">Dark</span>
          </label>

          <label
            className={cn(
              'flex-1 p-3 rounded-kiln text-center opacity-40 cursor-not-allowed',
              'bg-kiln-900 border border-kiln-600',
            )}
            title="Light theme coming soon"
          >
            <input
              type="radio"
              name="theme"
              value="light"
              disabled
              className="sr-only"
            />
            <div className="w-5 h-5 mx-auto mb-2 rounded-full bg-kiln-300/40 border border-kiln-400/30" />
            <span className="text-xs font-medium text-kiln-500">Light (soon)</span>
          </label>
        </div>
      </SettingField>
    </SectionCard>
  );
}

function QuarrySection({ settings, update }: SectionProps) {
  const strategies = [
    { value: 'hierarchy', label: 'Hierarchy', description: 'Structure-aware chunking based on headings' },
    { value: 'recursive', label: 'Recursive', description: 'Natural boundary splitting (paragraphs, sentences)' },
    { value: 'fixed', label: 'Fixed Size', description: 'Simple fixed-size chunks with overlap' },
  ];

  return (
    <SectionCard id="quarry" title="Quarry" icon={Mountain} iconColor="#7C92A8">
      <SettingField
        label="Default Processing Strategy"
        description="The chunking strategy applied when processing new documents."
      >
        <div className="flex items-center gap-2">
          <Layers size={14} className="text-kiln-500 flex-shrink-0" />
          <select
            value={settings.quarryDefaultStrategy}
            onChange={(e) => update('quarryDefaultStrategy', e.target.value)}
            className="input-field"
          >
            {strategies.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label} -- {s.description}
              </option>
            ))}
          </select>
        </div>
      </SettingField>

      <SettingField
        label="Max File Size (MB)"
        description="Maximum PDF file size accepted for processing. Default 100 MB."
      >
        <div className="flex items-center gap-2">
          <FileBox size={14} className="text-kiln-500 flex-shrink-0" />
          <input
            type="number"
            value={settings.quarryMaxFileSizeMb}
            onChange={(e) => update('quarryMaxFileSizeMb', Math.max(1, parseInt(e.target.value) || 100))}
            min={1}
            max={1000}
            className="input-field w-32"
          />
          <span className="text-2xs text-kiln-500">MB</span>
        </div>
      </SettingField>
    </SectionCard>
  );
}

function ForgeSection({ settings, update }: SectionProps) {
  return (
    <SectionCard id="forge" title="Forge" icon={Hammer} iconColor="#D4915C">
      <SettingField
        label="Target Examples per Competency"
        description="The goal number of training examples to collect for each competency. Recommended 50-100."
      >
        <div className="flex items-center gap-2">
          <Target size={14} className="text-kiln-500 flex-shrink-0" />
          <input
            type="number"
            value={settings.forgeTargetExamples}
            onChange={(e) => update('forgeTargetExamples', Math.max(5, parseInt(e.target.value) || 50))}
            min={5}
            max={500}
            className="input-field w-32"
          />
          <span className="text-2xs text-kiln-500">examples</span>
        </div>
      </SettingField>

      <ToggleSwitch
        checked={settings.forgeAutoSave}
        onChange={(v) => update('forgeAutoSave', v)}
        label="Auto-save"
        description="Automatically save discipline and example changes as you work."
        accentColor="#D4915C"
      />
    </SectionCard>
  );
}

function FoundrySection({ settings, update }: SectionProps) {
  const models = [
    { value: 'meta-llama/Llama-3.1-8B', label: 'Llama 3.1 8B' },
    { value: 'mistralai/Mistral-7B-v0.3', label: 'Mistral 7B v0.3' },
    { value: 'microsoft/phi-3-mini', label: 'Phi-3 Mini (3.8B)' },
  ];

  return (
    <SectionCard id="foundry" title="Foundry" icon={Factory} iconColor="#6BA089">
      <SettingField
        label="Default Base Model"
        description="The pre-trained model used as the starting point for LoRA fine-tuning."
      >
        <div className="flex items-center gap-2">
          <Cpu size={14} className="text-kiln-500 flex-shrink-0" />
          <select
            value={settings.foundryBaseModel}
            onChange={(e) => update('foundryBaseModel', e.target.value)}
            className="input-field"
          >
            {models.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
      </SettingField>

      <SettingField
        label="Default LoRA Rank"
        description="Rank of the low-rank adaptation matrices. Higher = more expressive, slower. Common values: 8, 16, 32, 64."
      >
        <div className="flex items-center gap-2">
          <Hash size={14} className="text-kiln-500 flex-shrink-0" />
          <input
            type="number"
            value={settings.foundryLoraRank}
            onChange={(e) => update('foundryLoraRank', Math.max(4, parseInt(e.target.value) || 16))}
            min={4}
            max={128}
            step={4}
            className="input-field w-32"
          />
        </div>
      </SettingField>
    </SectionCard>
  );
}

function HearthSection({ settings, update }: SectionProps) {
  const slots = [
    { value: 'slot-0', label: 'Slot 0 (Primary)' },
    { value: 'slot-1', label: 'Slot 1' },
    { value: 'slot-2', label: 'Slot 2' },
  ];

  return (
    <SectionCard id="hearth" title="Hearth" icon={Flame} iconColor="#D4A058">
      <SettingField
        label="Default Model Slot"
        description="Which model slot to use when starting a new conversation."
      >
        <div className="flex items-center gap-2">
          <MessageSquare size={14} className="text-kiln-500 flex-shrink-0" />
          <select
            value={settings.hearthDefaultSlot}
            onChange={(e) => update('hearthDefaultSlot', e.target.value)}
            className="input-field"
          >
            {slots.map((s) => (
              <option key={s.value} value={s.value}>
                {s.label}
              </option>
            ))}
          </select>
        </div>
      </SettingField>

      <ToggleSwitch
        checked={settings.hearthStreaming}
        onChange={(v) => update('hearthStreaming', v)}
        label="Streaming responses"
        description="Stream tokens as they are generated for a faster-feeling experience."
        accentColor="#D4A058"
      />
    </SectionCard>
  );
}

function AboutSection() {
  const toolInfo = [
    { name: 'Quarry', icon: Mountain, color: '#7C92A8', status: 'Complete', tests: 569 },
    { name: 'Forge', icon: Hammer, color: '#D4915C', status: 'Complete', tests: 400 },
    { name: 'Foundry', icon: Factory, color: '#6BA089', status: 'Complete', tests: 406 },
    { name: 'Hearth', icon: Flame, color: '#D4A058', status: 'Complete', tests: 144 },
  ];

  const links = [
    { label: 'Architecture', href: '#docs-architecture', icon: BookOpen },
    { label: 'API Reference', href: '#docs-api', icon: BookOpen },
    { label: 'Installation', href: '#docs-install', icon: BookOpen },
    { label: 'Troubleshooting', href: '#docs-troubleshooting', icon: BookOpen },
  ];

  return (
    <SectionCard id="about" title="About" icon={Info} iconColor="#8892A8">
      {/* App identity */}
      <div className="flex items-start gap-4">
        <div
          className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{ background: 'rgba(232, 115, 74, 0.10)' }}
        >
          <Zap size={22} className="text-ember" />
        </div>
        <div>
          <h4 className="font-display text-base font-semibold text-kiln-100">
            Kiln
          </h4>
          <p className="text-xs text-kiln-400 mt-0.5">
            Version 0.2.0 (MVP)
          </p>
          <p className="text-xs text-kiln-500 mt-2 leading-relaxed max-w-md">
            A complete pipeline for trustworthy domain-specific AI.
            Process documents, build curricula, train models, and
            query with citations -- all running locally.
          </p>
        </div>
      </div>

      {/* Tool status grid */}
      <div>
        <h4 className="text-xs font-medium text-kiln-400 uppercase tracking-wide mb-3">
          Tool Status
        </h4>
        <div className="grid grid-cols-2 gap-2">
          {toolInfo.map((tool) => {
            const Icon = tool.icon;
            return (
              <div
                key={tool.name}
                className="flex items-center gap-3 px-3 py-2.5 rounded-kiln bg-kiln-900 border border-kiln-600/50"
              >
                <Icon size={14} style={{ color: tool.color }} />
                <div className="flex-1 min-w-0">
                  <span className="text-xs font-medium text-kiln-200">{tool.name}</span>
                  <div className="flex items-center gap-2 mt-0.5">
                    <span className="text-2xs text-success">{tool.status}</span>
                    <span className="text-2xs text-kiln-600">|</span>
                    <span className="text-2xs text-kiln-500">{tool.tests} tests</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Documentation links */}
      <div>
        <h4 className="text-xs font-medium text-kiln-400 uppercase tracking-wide mb-3">
          Documentation
        </h4>
        <div className="grid grid-cols-2 gap-2">
          {links.map((link) => {
            const LinkIcon = link.icon;
            return (
              <a
                key={link.label}
                href={link.href}
                className={cn(
                  'flex items-center gap-2 px-3 py-2 rounded-kiln text-xs text-kiln-300',
                  'bg-kiln-900 border border-kiln-600/50',
                  'hover:bg-kiln-700 hover:text-kiln-100 transition-colors duration-150',
                )}
              >
                <LinkIcon size={13} className="text-kiln-500" />
                {link.label}
                <ExternalLink size={10} className="text-kiln-600 ml-auto" />
              </a>
            );
          })}
        </div>
      </div>

      {/* Footer line */}
      <p className="text-2xs text-kiln-600 pt-2">
        Built with FastAPI, React, Electron, and Tailwind CSS.
      </p>
    </SectionCard>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Component                                                     */
/* ------------------------------------------------------------------ */

export function SettingsPage() {
  const [settings, setSettings] = useState<KilnSettings>(DEFAULT_SETTINGS);
  const [activeSection, setActiveSection] = useState<SectionId>('general');
  const [hasChanges, setHasChanges] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const isScrollingByClick = useRef(false);

  const update = <K extends keyof KilnSettings>(key: K, value: KilnSettings[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleReset = () => {
    setSettings(DEFAULT_SETTINGS);
    setHasChanges(true);
  };

  const handleSave = async () => {
    setIsSaving(true);
    // Simulate saving (integrate with real API when available)
    await new Promise((resolve) => setTimeout(resolve, 400));
    setHasChanges(false);
    setIsSaving(false);
  };

  const scrollToSection = (id: SectionId) => {
    const el = document.getElementById(`section-${id}`);
    if (el && scrollRef.current) {
      isScrollingByClick.current = true;
      setActiveSection(id);
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      // Reset click-scrolling flag after the scroll animation settles
      setTimeout(() => {
        isScrollingByClick.current = false;
      }, 600);
    }
  };

  // Track which section is in view when the user scrolls naturally
  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;

    const handleScroll = () => {
      if (isScrollingByClick.current) return;

      const sectionIds: SectionId[] = ['general', 'quarry', 'forge', 'foundry', 'hearth', 'about'];
      let closestId: SectionId = 'general';
      let closestDistance = Infinity;

      for (const id of sectionIds) {
        const el = document.getElementById(`section-${id}`);
        if (el) {
          const rect = el.getBoundingClientRect();
          const containerRect = container.getBoundingClientRect();
          const distance = Math.abs(rect.top - containerRect.top - 24);
          if (distance < closestDistance) {
            closestDistance = distance;
            closestId = id;
          }
        }
      }

      setActiveSection(closestId);
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="flex flex-col h-full">
      <ToolHeader
        icon={Settings}
        title="Settings"
        color="#8892A8"
      >
        {hasChanges && (
          <div className="flex items-center gap-2">
            <button
              onClick={handleReset}
              className="btn-ghost btn-sm"
              disabled={isSaving}
            >
              <RotateCcw size={13} />
              Reset
            </button>
            <button
              onClick={handleSave}
              className="btn-primary btn-sm"
              disabled={isSaving}
            >
              <Save size={13} />
              {isSaving ? 'Saving...' : 'Save'}
            </button>
          </div>
        )}
      </ToolHeader>

      <div className="flex-1 flex overflow-hidden">
        {/* Left nav */}
        <nav className="w-48 flex-shrink-0 border-r border-kiln-600 bg-kiln-800/30 py-4 px-3">
          <div className="space-y-0.5">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeSection === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => scrollToSection(item.id)}
                  className={cn(
                    'w-full flex items-center gap-2.5 px-3 py-2 rounded-kiln text-xs font-medium',
                    'transition-all duration-150 text-left',
                    isActive
                      ? 'bg-kiln-700 text-kiln-100'
                      : 'text-kiln-400 hover:text-kiln-200 hover:bg-kiln-700/50',
                  )}
                >
                  <Icon
                    size={14}
                    style={{ color: isActive ? item.color : undefined }}
                    className={cn(!isActive && 'text-kiln-500')}
                  />
                  {item.label}
                </button>
              );
            })}
          </div>

          {/* Unsaved changes indicator */}
          {hasChanges && (
            <div className="mt-6 mx-3 px-3 py-2 rounded-kiln bg-warning/10 border border-warning/20">
              <p className="text-2xs text-warning font-medium">Unsaved changes</p>
              <p className="text-2xs text-kiln-500 mt-0.5">
                Press Save in the header to apply.
              </p>
            </div>
          )}
        </nav>

        {/* Scrollable content area */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto p-6"
        >
          <div className="max-w-2xl mx-auto space-y-5">
            <GeneralSection settings={settings} update={update} />
            <QuarrySection settings={settings} update={update} />
            <ForgeSection settings={settings} update={update} />
            <FoundrySection settings={settings} update={update} />
            <HearthSection settings={settings} update={update} />
            <AboutSection />

            {/* Bottom padding so last section can scroll to top */}
            <div className="h-32" aria-hidden="true" />
          </div>
        </div>
      </div>
    </div>
  );
}
