import React, { useState } from 'react';
import { Upload, Youtube, Plus, X, FileType, Loader2, Music } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface UploadFormProps {
  onUpload: (ppt: File | null, links: string[], audio: File | null) => Promise<void> | void;
}

export function UploadForm({ onUpload }: UploadFormProps) {
  const [ppt, setPpt] = useState<File | null>(null);
  const [audio, setAudio] = useState<File | null>(null);
  const [links, setLinks] = useState<string[]>(['']);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleAddLink = () => setLinks([...links, '']);
  const handleRemoveLink = (index: number) => {
    const newLinks = links.filter((_, i) => i !== index);
    setLinks(newLinks.length ? newLinks : ['']);
  };

  const handleLinkChange = (index: number, value: string) => {
    const newLinks = [...links];
    newLinks[index] = value;
    setLinks(newLinks);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (isSubmitting) return;
    const cleanedLinks = links.filter(l => l.trim() !== '');
    if (!audio && cleanedLinks.length === 0) {
      alert("Please provide either an audio file or at least one YouTube link.");
      return;
    }
    setIsSubmitting(true);
    await onUpload(ppt, cleanedLinks, audio);
    setIsSubmitting(false);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-2xl mx-auto p-8 bg-white rounded-3xl shadow-xl border border-black/5"
    >
      <h2 className="text-3xl font-bold tracking-tight text-zinc-900 mb-2">Create Study Material</h2>
      <p className="text-zinc-500 mb-8">Upload slides, then choose either an audio file or YouTube links.</p>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* PPT Upload */}
        <div className="space-y-3">
          <label className="text-sm font-semibold uppercase tracking-wider text-zinc-400">Lecture Slides (PPT/PDF)</label>
          <div
            className={`relative border-2 border-dashed rounded-2xl p-8 transition-all ${
              ppt ? 'border-emerald-500 bg-emerald-50/50' : 'border-zinc-200 hover:border-zinc-300'
            }`}
          >
            <input
              type="file"
              accept=".ppt,.pptx,.pdf"
              onChange={(e) => setPpt(e.target.files?.[0] || null)}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <div className="flex flex-col items-center justify-center text-center">
              {ppt ? (
                <>
                  <FileType className="w-12 h-12 text-emerald-500 mb-3" />
                  <p className="font-medium text-zinc-900">{ppt.name}</p>
                  <p className="text-sm text-zinc-500">{(ppt.size / 1024 / 1024).toFixed(2)} MB</p>
                </>
              ) : (
                <>
                  <Upload className="w-12 h-12 text-zinc-300 mb-3" />
                  <p className="font-medium text-zinc-900">Click or drag to upload</p>
                  <p className="text-sm text-zinc-500">Supports PPT, PPTX, and PDF</p>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Audio Upload (Optional alternative to links) */}
        <div className="space-y-3">
          <label className="text-sm font-semibold uppercase tracking-wider text-zinc-400">Lecture Audio (Optional)</label>
          <div
            className={`relative border-2 border-dashed rounded-2xl p-8 transition-all ${
              audio ? 'border-emerald-500 bg-emerald-50/50' : 'border-zinc-200 hover:border-zinc-300'
            }`}
          >
            <input
              type="file"
              accept=".wav,.mp3,.webm,.m4a,.aac,.flac,.ogg"
              onChange={(e) => setAudio(e.target.files?.[0] || null)}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <div className="flex flex-col items-center justify-center text-center">
              {audio ? (
                <>
                  <Music className="w-12 h-12 text-emerald-500 mb-3" />
                  <p className="font-medium text-zinc-900">{audio.name}</p>
                  <p className="text-sm text-zinc-500">{(audio.size / 1024 / 1024).toFixed(2)} MB</p>
                </>
              ) : (
                <>
                  <Music className="w-12 h-12 text-zinc-300 mb-3" />
                  <p className="font-medium text-zinc-900">Click or drag to upload audio</p>
                  <p className="text-sm text-zinc-500">If audio is uploaded, YouTube links are optional</p>
                </>
              )}
            </div>
          </div>
        </div>

        {/* YouTube Links */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-semibold uppercase tracking-wider text-zinc-400">YouTube Links</label>
            <button
              type="button"
              onClick={handleAddLink}
              className="text-xs font-bold text-emerald-600 hover:text-emerald-700 flex items-center gap-1"
            >
              <Plus className="w-3 h-3" /> Add Link
            </button>
          </div>
          <div className="space-y-3">
            <AnimatePresence initial={false}>
              {links.map((link, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="flex gap-2"
                >
                  <div className="relative flex-1">
                    <Youtube className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-400" />
                    <input
                      type="url"
                      value={link}
                      onChange={(e) => handleLinkChange(index, e.target.value)}
                      placeholder="https://youtube.com/watch?v=..."
                      className="w-full pl-11 pr-4 py-3 bg-zinc-50 border border-zinc-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all"
                    />
                  </div>
                  {links.length > 1 && (
                    <button
                      type="button"
                      onClick={() => handleRemoveLink(index)}
                      className="p-3 text-zinc-400 hover:text-red-500 transition-colors"
                    >
                      <X className="w-5 h-5" />
                    </button>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        <button
          type="submit"
          disabled={isSubmitting}
          className="w-full py-4 bg-zinc-900 text-white font-bold rounded-2xl hover:bg-zinc-800 transition-all shadow-lg shadow-zinc-900/10 active:scale-[0.98] disabled:opacity-70 disabled:cursor-not-allowed flex flex-col items-center justify-center gap-1"
        >
          {isSubmitting ? (
            <>
              <div className="flex items-center gap-2">
                <Loader2 className="w-5 h-5 animate-spin" />
                Processing...
              </div>
              <span className="text-xs text-zinc-400 font-normal mt-0.5">This may take 1-3 minutes</span>
            </>
          ) : (
            'Generate Study Material'
          )}
        </button>
      </form>
    </motion.div>
  );
}
