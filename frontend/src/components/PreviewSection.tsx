import { useState, useEffect, useRef } from 'react';
import { documentRedactionAPI, DocumentPreviewResponse, RedactedPreviewResponse } from '@/services/api';
import { AlertCircle, ChevronLeft, ChevronRight, Download, Maximize2, X } from 'lucide-react';
import { Panel } from './ui/panel';
import { Badge } from './ui/badge';
import { Button } from './ui/button';

interface PreviewSectionProps {
  documentId?: string;
  summary: {
    entitiesRedacted: number;
    avgConfidence: number;
  };
  refreshTrigger?: number;
}

type FileKind = 'pdf' | 'image' | 'text';

function fileKind(filename: string): FileKind {
  const name = filename.toLowerCase();
  if (name.endsWith('.pdf')) return 'pdf';
  if (/\.(jpg|jpeg|png|gif|bmp|webp|tiff)$/.test(name)) return 'image';
  return 'text';
}

/**
 * One page at a time, rendered to PNG by the server - so what the reviewer
 * sees is exactly the rasterised output file, identical on every browser
 * (iframe'd PDF blobs render poorly or not at all on mobile Safari).
 */
function PageViewer({
  documentId,
  redacted,
  refreshKey,
  onExpand,
}: {
  documentId: string;
  redacted: boolean;
  refreshKey?: number;
  onExpand: (url: string) => void;
}) {
  const [page, setPage] = useState(1);
  const [pageCount, setPageCount] = useState(1);
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const urlRef = useRef<string | null>(null);

  useEffect(() => setPage(1), [documentId, refreshKey]);

  useEffect(() => {
    let cancelled = false;
    setFailed(false);
    documentRedactionAPI
      .getPageImage(documentId, page, redacted)
      .then(({ url: next, pageCount: count }) => {
        if (cancelled) return URL.revokeObjectURL(next);
        if (urlRef.current) URL.revokeObjectURL(urlRef.current);
        urlRef.current = next;
        setUrl(next);
        setPageCount(count);
      })
      .catch(() => !cancelled && setFailed(true));
    return () => {
      cancelled = true;
    };
  }, [documentId, page, redacted, refreshKey]);

  useEffect(() => () => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
  }, []);

  if (failed) {
    return (
      <div className="flex flex-1 items-center justify-center border border-line bg-paper px-4 text-center font-display text-[11px] uppercase tracking-[0.08em] text-ink-faint">
        Preview unavailable
      </div>
    );
  }
  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <div className="group relative flex-1 overflow-auto border border-line bg-paper">
        {url ? (
          <img src={url} alt={`${redacted ? 'Redacted' : 'Original'} page ${page}`} className="mx-auto w-full" />
        ) : (
          <div className="flex h-full items-center justify-center font-display text-[11px] uppercase tracking-[0.08em] text-ink-faint">
            Rendering page…
          </div>
        )}
        {url && (
          <button
            onClick={() => onExpand(url)}
            className="absolute right-2 top-2 flex h-8 w-8 items-center justify-center border border-ink bg-paper-raised/90 text-ink opacity-100 transition-colors hover:bg-ink hover:text-paper-raised sm:opacity-0 sm:group-hover:opacity-100 sm:focus-visible:opacity-100"
            aria-label="Open page fullscreen"
          >
            <Maximize2 className="h-4 w-4" />
          </button>
        )}
      </div>
      {pageCount > 1 && (
        <div className="mt-2 flex items-center justify-between font-display text-[11px] uppercase tracking-[0.06em] text-ink-faint">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="flex items-center gap-1 disabled:opacity-40"
            aria-label="Previous page"
          >
            <ChevronLeft className="h-3.5 w-3.5" /> Prev
          </button>
          <span>
            Page {page} / {pageCount}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(pageCount, p + 1))}
            disabled={page >= pageCount}
            className="flex items-center gap-1 disabled:opacity-40"
            aria-label="Next page"
          >
            Next <ChevronRight className="h-3.5 w-3.5" />
          </button>
        </div>
      )}
    </div>
  );
}

function TextPreview({ text }: { text: string }) {
  return (
    <pre className="flex-1 overflow-auto whitespace-pre-wrap border border-line bg-paper p-4 font-serif text-sm leading-relaxed text-ink">
      {text || 'No text extracted.'}
    </pre>
  );
}

export default function PreviewSection({ documentId, summary, refreshTrigger }: PreviewSectionProps) {
  const [documentPreview, setDocumentPreview] = useState<DocumentPreviewResponse | null>(null);
  const [redactedPreview, setRedactedPreview] = useState<RedactedPreviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [redactedUrl, setRedactedUrl] = useState<string | null>(null);
  const [originalError, setOriginalError] = useState(false);
  const [redactedError, setRedactedError] = useState(false);

  const [fullscreen, setFullscreen] = useState<{ url: string; label: string } | null>(null);

  const originalRef = useRef<string | null>(null);
  const redactedRef = useRef<string | null>(null);

  const putOriginal = (url: string | null) => {
    if (originalRef.current) URL.revokeObjectURL(originalRef.current);
    originalRef.current = url;
    setOriginalUrl(url);
  };
  const putRedacted = (url: string | null) => {
    if (redactedRef.current) URL.revokeObjectURL(redactedRef.current);
    redactedRef.current = url;
    setRedactedUrl(url);
  };

  useEffect(() => {
    if (!documentId) return;
    let cancelled = false;

    putOriginal(null);
    putRedacted(null);
    setRedactedPreview(null);
    setOriginalError(false);
    setRedactedError(false);

    (async () => {
      setLoading(true);
      setError(null);
      try {
        const preview = await documentRedactionAPI.getDocumentPreview(documentId);
        if (cancelled) return;
        setDocumentPreview(preview);
        const kind = fileKind(preview.filename);

        if (kind === 'image') {
          documentRedactionAPI
            .getDocumentUrl(documentId, false)
            .then((url) => (cancelled ? URL.revokeObjectURL(url) : putOriginal(url)))
            .catch(() => !cancelled && setOriginalError(true));
        }

        if (preview.status === 'redacted') {
          try {
            const redacted = await documentRedactionAPI.getRedactedPreview(documentId);
            if (cancelled) return;
            setRedactedPreview(redacted);
            if (kind === 'image') {
              documentRedactionAPI
                .getDocumentUrl(documentId, true)
                .then((url) => (cancelled ? URL.revokeObjectURL(url) : putRedacted(url)))
                .catch(() => !cancelled && setRedactedError(true));
            }
          } catch {
            // redacted preview not ready yet — original still renders
          }
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load document data');
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
     
  }, [documentId, refreshTrigger]);

  useEffect(
    () => () => {
      if (originalRef.current) URL.revokeObjectURL(originalRef.current);
      if (redactedRef.current) URL.revokeObjectURL(redactedRef.current);
    },
    [],
  );

  const handleDownload = async () => {
    if (!documentId) return;
    try {
      const preview = redactedPreview || documentPreview;
      if (preview) await documentRedactionAPI.downloadAndSaveDocument(documentId, preview.filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Download failed');
    }
  };

  const kind = fileKind(documentPreview?.filename ?? '');

  const renderViewer = (url: string, label: string) => (
    <img src={url} alt={label} className="h-full w-full object-contain" />
  );

  const renderExhibit = (redacted: boolean) => {
    if (!documentId) return null;
    if (kind === 'pdf') {
      return (
        <PageViewer
          documentId={documentId}
          redacted={redacted}
          refreshKey={refreshTrigger}
          onExpand={(url) => setFullscreen({ url, label: redacted ? 'Redacted document' : 'Original document' })}
        />
      );
    }
    if (kind === 'text') {
      return (
        <TextPreview
          text={(redacted ? redactedPreview?.redacted_text_preview : documentPreview?.extracted_text_preview) ?? ''}
        />
      );
    }
    return redacted
      ? renderInline(redactedUrl, redactedError, 'Redacted document')
      : renderInline(originalUrl, originalError, 'Original document');
  };

  const renderInline = (url: string | null, failed: boolean, label: string) => {
    if (failed) {
      return (
        <div className="flex flex-1 items-center justify-center border border-line bg-paper px-4 text-center font-display text-[11px] uppercase tracking-[0.08em] text-ink-faint">
          Preview unavailable
        </div>
      );
    }
    if (!url) {
      return (
        <div className="flex flex-1 items-center justify-center border border-line bg-paper font-display text-[11px] uppercase tracking-[0.08em] text-ink-faint">
          Loading preview…
        </div>
      );
    }
    return (
      <div className="group relative flex-1 overflow-hidden border border-line bg-paper">
        {renderViewer(url, label)}
        <button
          onClick={() => setFullscreen({ url, label })}
          className="absolute right-2 top-2 flex h-8 w-8 items-center justify-center border border-ink bg-paper-raised/90 text-ink opacity-100 transition-colors hover:bg-ink hover:text-paper-raised sm:opacity-0 sm:group-hover:opacity-100 sm:focus-visible:opacity-100"
          aria-label={`Open ${label} fullscreen`}
        >
          <Maximize2 className="h-4 w-4" />
        </button>
      </div>
    );
  };

  if (loading && !documentPreview) {
    return (
      <div className="flex h-48 items-center justify-center border border-line bg-paper-raised">
        <span className="font-display text-xs uppercase tracking-[0.08em] text-ink-faint">
          Loading exhibit…
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-start gap-3 border border-alert bg-alert-soft/40 p-5 sm:p-6">
        <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-alert" />
        <div>
          <p className="font-display text-sm font-semibold text-alert">Couldn't load this document</p>
          <p className="mt-1 font-serif text-sm text-ink-soft">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 sm:space-y-8">
      <div className="grid grid-cols-3 divide-x divide-line border border-line bg-paper-raised">
        <div className="p-3 text-center sm:p-6">
          <p className="font-display text-2xl font-semibold text-ink sm:text-3xl">
            {documentPreview?.entity_count ?? 0}
          </p>
          <p className="mt-1 font-display text-[9px] uppercase leading-tight tracking-[0.06em] text-ink-faint sm:text-[10px] sm:tracking-[0.08em]">
            Entities detected
          </p>
        </div>
        <div className="p-3 text-center sm:p-6">
          <p className="font-display text-2xl font-semibold text-flag sm:text-3xl">{summary.entitiesRedacted}</p>
          <p className="mt-1 font-display text-[9px] uppercase leading-tight tracking-[0.06em] text-ink-faint sm:text-[10px] sm:tracking-[0.08em]">
            Entities redacted
          </p>
        </div>
        <div className="p-3 text-center sm:p-6">
          <p className="font-display text-2xl font-semibold text-confirm sm:text-3xl">{summary.avgConfidence}%</p>
          <p className="mt-1 font-display text-[9px] uppercase leading-tight tracking-[0.06em] text-ink-faint sm:text-[10px] sm:tracking-[0.08em]">
            Average confidence
          </p>
        </div>
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        <Panel tab="EXHIBIT A — ORIGINAL" className="flex h-[26rem] flex-col p-4 sm:h-[28rem] sm:p-5">
          {documentPreview ? (
            <>
              <div className="flex items-center justify-between gap-3 border-b border-line pb-3">
                <span className="truncate font-display text-xs font-medium text-ink">
                  {documentPreview.filename}
                </span>
                <Badge tone="muted">{documentPreview.entity_count} found</Badge>
              </div>
              {renderExhibit(false)}
            </>
          ) : (
            <div className="flex flex-1 items-center justify-center font-display text-xs uppercase tracking-[0.08em] text-ink-faint">
              No document loaded
            </div>
          )}
        </Panel>

        <Panel tab="EXHIBIT B — REDACTED" className="flex h-[26rem] flex-col p-4 sm:h-[28rem] sm:p-5">
          {redactedPreview ? (
            <>
              <div className="flex items-center justify-between gap-3 border-b border-line pb-3">
                <span className="truncate font-display text-xs font-medium text-ink">
                  {redactedPreview.redacted_filename}
                </span>
                <Badge tone="flag">{redactedPreview.redacted_count} redacted</Badge>
              </div>
              {renderExhibit(true)}
              <Button size="sm" className="mt-3 w-full" onClick={handleDownload}>
                <Download className="h-3.5 w-3.5" />
                Download
              </Button>
            </>
          ) : documentPreview?.status === 'redacted' ? (
            <div className="flex flex-1 items-center justify-center font-display text-xs uppercase tracking-[0.08em] text-ink-faint">
              Loading redacted exhibit…
            </div>
          ) : (
            <div className="flex flex-1 items-center justify-center font-display text-xs uppercase tracking-[0.08em] text-ink-faint">
              Not redacted yet
            </div>
          )}
        </Panel>
      </div>

      {fullscreen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-ink/80 p-3 sm:p-4"
          onClick={() => setFullscreen(null)}
        >
          <div
            className="flex h-[90vh] w-full max-w-6xl flex-col border border-ink bg-paper-raised"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-line p-3 sm:p-4">
              <span className="truncate font-display text-sm font-semibold text-ink">{fullscreen.label}</span>
              <button
                onClick={() => setFullscreen(null)}
                className="text-ink-faint hover:text-ink"
                aria-label="Close viewer"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="flex-1 p-3 sm:p-4">{renderViewer(fullscreen.url, fullscreen.label)}</div>
          </div>
        </div>
      )}
    </div>
  );
}
