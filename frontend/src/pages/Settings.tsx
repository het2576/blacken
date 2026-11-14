import { useEffect, useState } from 'react';
import { AlertCircle, CheckCircle2 } from 'lucide-react';
import { documentRedactionAPI, ConfigResponse, HealthResponse, Preferences } from '@/services/api';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Panel } from '@/components/ui/panel';

function NumberField({
  label,
  hint,
  value,
  onChange,
  min,
  max,
  step,
  suffix,
}: {
  label: string;
  hint: string;
  value: string;
  onChange: (v: string) => void;
  min: number;
  max: number;
  step: number;
  suffix: string;
}) {
  return (
    <div>
      <label className="font-display text-[11px] font-medium uppercase tracking-[0.08em] text-ink">
        {label}
      </label>
      <p className="mt-1 font-serif text-sm text-ink-soft">{hint}</p>
      <div className="mt-3 flex items-center gap-2">
        <input
          type="number"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={(e) => onChange(e.target.value)}
          className="w-28 border border-line bg-paper px-3 py-2 font-display text-sm text-ink focus-visible:border-ink"
        />
        <span className="font-display text-xs uppercase tracking-[0.06em] text-ink-faint">{suffix}</span>
      </div>
    </div>
  );
}

const toLines = (terms: string[]) => terms.join('\n');
const fromLines = (text: string) =>
  text
    .split('\n')
    .map((t) => t.trim())
    .filter(Boolean);

function TermsField({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="font-display text-[11px] font-medium uppercase tracking-[0.08em] text-ink">{label}</label>
      <p className="mt-1 font-serif text-sm text-ink-soft">{hint}</p>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        rows={6}
        placeholder="One term per line"
        className="mt-3 w-full border border-line bg-paper px-3 py-2 font-serif text-sm text-ink focus-visible:border-ink"
      />
    </div>
  );
}

function RedactionRules() {
  const [saved, setSaved] = useState<Preferences>({ always_redact: [], never_flag: [] });
  const [always, setAlways] = useState('');
  const [never, setNever] = useState('');
  const [status, setStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    documentRedactionAPI
      .getPreferences()
      .then((prefs) => {
        setSaved(prefs);
        setAlways(toLines(prefs.always_redact));
        setNever(toLines(prefs.never_flag));
      })
      .catch(() => setMessage('Could not load your rules.'));
  }, []);

  const dirty = toLines(fromLines(always)) !== toLines(saved.always_redact) || toLines(fromLines(never)) !== toLines(saved.never_flag);

  const save = async () => {
    setStatus('saving');
    setMessage(null);
    try {
      const prefs = await documentRedactionAPI.savePreferences({
        always_redact: fromLines(always),
        never_flag: fromLines(never),
      });
      setSaved(prefs);
      setAlways(toLines(prefs.always_redact));
      setNever(toLines(prefs.never_flag));
      setStatus('saved');
      setTimeout(() => setStatus('idle'), 3000);
    } catch (err) {
      setStatus('error');
      setMessage(err instanceof Error ? err.message : 'Failed to save');
    }
  };

  return (
    <Panel tab="YOUR REDACTION RULES — THIS BROWSER" className="p-6 sm:p-8">
      <div className="grid gap-8 sm:grid-cols-2">
        <TermsField
          label="Always redact"
          hint="Names or codes the detector might not know — a client, a project codename. Every occurrence is pre-ticked."
          value={always}
          onChange={setAlways}
        />
        <TermsField
          label="Never flag"
          hint="Terms that are never sensitive for you, such as your own company name. They won't be suggested."
          value={never}
          onChange={setNever}
        />
      </div>
      <div className="mt-6 flex flex-wrap items-center gap-4 border-t border-line pt-6">
        <Button variant="flag" onClick={save} disabled={!dirty || status === 'saving'}>
          {status === 'saving' ? 'Saving…' : 'Save rules'}
        </Button>
        {status === 'saved' && (
          <span className="flex items-center gap-1.5 font-display text-[11px] uppercase tracking-[0.06em] text-confirm">
            <CheckCircle2 className="h-3.5 w-3.5" />
            Saved
          </span>
        )}
        {message && <span className="font-display text-[11px] uppercase tracking-[0.06em] text-alert">{message}</span>}
        <span className="font-serif text-sm text-ink-faint">
          Applied to documents you upload from now on. Stored encrypted on the server and tied to this browser.
        </span>
      </div>
    </Panel>
  );
}

export default function Settings() {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [retentionInput, setRetentionInput] = useState('');
  const [uploadLimitInput, setUploadLimitInput] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const [configRes, healthRes] = await Promise.all([
          documentRedactionAPI.getConfig(),
          documentRedactionAPI.checkHealth(),
        ]);
        setConfig(configRes);
        setHealth(healthRes);
        setRetentionInput(String(configRes.retention_hours));
        setUploadLimitInput(String(configRes.max_upload_mb));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load configuration');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setSaveError(null);
    setSaveSuccess(false);
    try {
      const updated = await documentRedactionAPI.updateConfig({
        retention_hours: Number(retentionInput),
        max_upload_mb: Number(uploadLimitInput),
      });
      setConfig(updated);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const dirty =
    config != null &&
    (Number(retentionInput) !== config.retention_hours || Number(uploadLimitInput) !== config.max_upload_mb);

  return (
    <div className="mx-auto max-w-5xl px-5 py-10 sm:px-8 sm:py-14">
      <span className="font-display text-[11px] font-medium uppercase tracking-[0.1em] text-ink-faint">
        Docket 04
      </span>
      <h1 className="mt-3 font-display text-display-2 font-semibold text-ink">Settings</h1>
      <p className="mt-3 max-w-lg font-serif text-lg leading-relaxed text-ink-soft">
        How this server is configured right now — most of it read-only by design.
      </p>

      {loading && (
        <p className="mt-10 border border-line bg-paper-raised p-8 text-center font-display text-xs uppercase tracking-[0.08em] text-ink-faint">
          Loading configuration…
        </p>
      )}

      {!loading && error && (
        <div className="mt-10 flex items-start gap-3 border border-alert bg-alert-soft/40 p-6">
          <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-alert" />
          <div>
            <p className="font-display text-sm font-semibold text-alert">Couldn't load configuration</p>
            <p className="mt-1 font-serif text-sm text-ink-soft">{error}</p>
          </div>
        </div>
      )}

      {!loading && !error && config && (
        <div className="mt-10 space-y-8">
          <RedactionRules />

          {config.config_editable && (
          <Panel tab="LIVE — TAKES EFFECT IMMEDIATELY" className="p-6 sm:p-8">
            <div className="grid gap-8 sm:grid-cols-2">
              <NumberField
                label="Retention window"
                hint="How long an uploaded document, its text, and its entities are kept before automatic deletion."
                value={retentionInput}
                onChange={setRetentionInput}
                min={0.1}
                max={720}
                step={0.5}
                suffix="hours"
              />
              <NumberField
                label="Max upload size"
                hint="Largest file the server will accept per upload."
                value={uploadLimitInput}
                onChange={setUploadLimitInput}
                min={0.1}
                max={100}
                step={1}
                suffix="MB"
              />
            </div>

            <div className="mt-6 flex flex-wrap items-center gap-4 border-t border-line pt-6">
              <Button variant="flag" onClick={handleSave} disabled={!dirty || saving}>
                {saving ? 'Saving…' : 'Save changes'}
              </Button>
              {saveSuccess && (
                <span className="flex items-center gap-1.5 font-display text-[11px] uppercase tracking-[0.06em] text-confirm">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Saved
                </span>
              )}
              {saveError && (
                <span className="font-display text-[11px] uppercase tracking-[0.06em] text-alert">{saveError}</span>
              )}
              <span className="font-serif text-sm text-ink-faint">
                These changes take effect right away, but they're temporary — the server restores its default values every time it restarts.
              </span>
            </div>
          </Panel>
          )}

          <Panel tab="READ-ONLY — SET VIA BACKEND/.ENV" className="p-6 sm:p-8">
            <div className="grid gap-6 sm:grid-cols-2">
              <div>
                <p className="font-display text-[11px] font-medium uppercase tracking-[0.08em] text-ink">
                  Detection engine
                </p>
                <p className="mt-2 font-serif text-[15px] text-ink-soft">
                  {config.spacy_model} · language: {config.detection_language}
                </p>
              </div>
              <div>
                <p className="font-display text-[11px] font-medium uppercase tracking-[0.08em] text-ink">
                  Access control
                </p>
                <p className="mt-2 font-serif text-[15px] text-ink-soft">
                  {config.api_key_required
                    ? 'An X-API-Key header is required on every request.'
                    : 'No API key is required — anyone who can reach this server can use it.'}
                </p>
              </div>
              <div>
                <p className="font-display text-[11px] font-medium uppercase tracking-[0.08em] text-ink">
                  Retention
                </p>
                <p className="mt-2 font-serif text-[15px] text-ink-soft">
                  Documents are deleted {config.retention_hours} hours after upload, or immediately when you delete them.
                </p>
              </div>
              <div>
                <p className="font-display text-[11px] font-medium uppercase tracking-[0.08em] text-ink">
                  Cleanup sweep
                </p>
                <p className="mt-2 font-serif text-[15px] text-ink-soft">
                  Runs every {config.cleanup_interval_minutes} minutes to remove anything past its retention window.
                </p>
              </div>
              <div>
                <p className="font-display text-[11px] font-medium uppercase tracking-[0.08em] text-ink">
                  Format support
                </p>
                <div className="mt-2 flex flex-wrap gap-2">
                  <Badge tone={health?.features.pdf ? 'confirm' : 'muted'}>PDF</Badge>
                  <Badge tone={health?.features.docx ? 'confirm' : 'muted'}>DOCX</Badge>
                  <Badge tone={health?.features.ocr ? 'confirm' : 'muted'}>OCR (scanned pages)</Badge>
                  <Badge tone={health?.features.transformer_model ? 'confirm' : 'muted'}>Transformer model</Badge>
                  <Badge tone={health?.features.encryption_at_rest ? 'confirm' : 'muted'}>Encrypted at rest</Badge>
                </div>
              </div>
            </div>

            <div className="mt-6 border-t border-line pt-6">
              <p className="font-display text-[11px] font-medium uppercase tracking-[0.08em] text-ink">
                Accepted file types
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                {config.allowed_extensions.map((ext) => (
                  <Badge key={ext} tone="muted">
                    {ext}
                  </Badge>
                ))}
              </div>
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}
