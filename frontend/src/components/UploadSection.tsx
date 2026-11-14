import { useCallback, useRef, useState } from 'react';
import { FileText, ImageIcon, UploadCloud, X } from 'lucide-react';
import { Button } from './ui/button';
import { Panel } from './ui/panel';
import { cn } from '@/lib/utils';

interface UploadSectionProps {
  onFileSelect: (file: File) => void;
  selectedFile: File | null;
}

const ACCEPTED = '.pdf,.docx,.txt,image/*';
const MAX_SIZE_MB = 20;

export default function UploadSection({ onFileSelect, selectedFile }: UploadSectionProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const openPicker = useCallback(() => inputRef.current?.click(), []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) onFileSelect(file);
    },
    [onFileSelect],
  );

  const isPdf = selectedFile?.name.toLowerCase().endsWith('.pdf');
  const fileSizeMb = selectedFile ? (selectedFile.size / 1024 / 1024).toFixed(2) : null;

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFileSelect(file);
        }}
      />

      {!selectedFile ? (
        <div
          onClick={openPicker}
          onDragOver={(e) => e.preventDefault()}
          onDragEnter={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            setIsDragging(false);
          }}
          onDrop={handleDrop}
          role="button"
          tabIndex={0}
          className={cn(
            'flex cursor-pointer flex-col items-center gap-5 border border-dashed border-line bg-paper-raised px-8 py-20 text-center transition-colors',
            isDragging && 'border-ink bg-flag-soft/40',
          )}
        >
          <span className="flex h-14 w-14 items-center justify-center border border-ink">
            <UploadCloud className="h-6 w-6 text-ink" strokeWidth={1.5} />
          </span>
          <div>
            <p className="font-display text-sm font-semibold uppercase tracking-[0.06em] text-ink">
              {isDragging ? 'Drop it' : 'Drop a document here'}
            </p>
            <p className="mt-2 font-serif text-[15px] text-ink-soft">
              or click to browse — PDF, DOCX, or an image, up to {MAX_SIZE_MB}MB
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              openPicker();
            }}
          >
            Browse files
          </Button>
        </div>
      ) : (
        <Panel tab="SELECTED" className="flex items-center justify-between gap-4 p-5">
          <div className="flex min-w-0 items-center gap-4">
            <span className="flex h-11 w-11 shrink-0 items-center justify-center border border-ink bg-ink">
              {isPdf ? (
                <FileText className="h-5 w-5 text-paper-raised" strokeWidth={1.5} />
              ) : (
                <ImageIcon className="h-5 w-5 text-paper-raised" strokeWidth={1.5} />
              )}
            </span>
            <div className="min-w-0">
              <p className="truncate font-display text-sm font-medium text-ink">
                {selectedFile.name}
              </p>
              <p className="font-display text-[11px] uppercase tracking-[0.06em] text-ink-faint">
                {fileSizeMb} MB
              </p>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Button variant="outline" size="sm" onClick={openPicker}>
              Change
            </Button>
            <button
              onClick={() => inputRef.current && (inputRef.current.value = '')}
              className="flex h-8 w-8 items-center justify-center text-ink-faint hover:text-ink"
              aria-label="Clear selection"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </Panel>
      )}
    </div>
  );
}
