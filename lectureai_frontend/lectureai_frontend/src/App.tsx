import { useState } from 'react';
import { UploadForm } from './components/UploadForm';
import { ProcessingView } from './components/ProcessingView';
import { Dashboard } from './components/Dashboard';
import { AppState, LectureResult } from './types';
import { Brain } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const normalizeNotes = (raw: any) => {
  if (!Array.isArray(raw)) return [];
  return raw.filter((n) => n && typeof n === 'object');
};

const normalizeFlashcards = (raw: any) => {
  if (Array.isArray(raw)) return raw;
  if (raw && typeof raw === 'object') {
    if (Array.isArray(raw.flashcards)) return raw.flashcards;
    return Object.values(raw).flatMap((item) => Array.isArray(item) ? item : [item]);
  }
  return [];
};

const normalizeQuestions = (raw: any) => {
  if (Array.isArray(raw)) {
    return raw;
  }
  if (raw && typeof raw === 'object') {
    if (Array.isArray(raw.questions)) {
      return raw.questions;
    }
    if (Array.isArray(raw.items)) {
      return raw.items;
    }
    return Object.values(raw).flatMap((item) => Array.isArray(item) ? item : [item]);
  }
  return [];
};

export default function App() {
  const [state, setState] = useState<AppState>('idle');
  const [result, setResult] = useState<LectureResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUpload = async (ppt: File | null, links: string[]) => {
    if (!ppt) {
      alert("Please upload a presentation file (PPT/PDF).");
      return;
    }
    
    setError(null);
    setState('processing');
    
    const formData = new FormData();
    formData.append('ppt', ppt);
    formData.append('links', JSON.stringify(links));

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 mins
      
      const response = await fetch(`${API_BASE_URL}/process_upload`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        let errorMsg = 'Something went wrong on the server.';
        try {
          const errData = await response.json();
          errorMsg = errData.detail || errorMsg;
        } catch(e) {}
        throw new Error(errorMsg);
      }
      
      const data = await response.json();
      
      const normalizedNotes = normalizeNotes(data.notes);
      const normalizedFlashcards = normalizeFlashcards(data.flashcards);
      const normalizedQuestions = normalizeQuestions(data.questions);

      const finalResult: LectureResult = {
        id: `lecture_${Date.now()}`,
        title: ppt.name,
        notes: normalizedNotes.map((n: any, idx: number) => ({
          slideNumber: idx + 1,
          title: `Slide: ${n.slide_id}`,
          content: String(n?.note || n?.text || '').trim()
        })),
        summary: data.summary || "No summary generated.",
        flashcards: normalizedFlashcards.map((f: any, i: number) => ({
          id: String(i),
          question: f?.q || f?.front || f?.question || "Question",
          answer: f?.a || f?.back || f?.answer || "Answer"
        })),
        questions: normalizedQuestions.map((q: any, i: number) => ({
          id: String(i),
          question: q?.q || q?.question || q?.prompt || `Question ${i + 1}`,
          answer: q?.answer || q?.answer_points || q?.expected_answer || "Answer"
        }))
      };
      
      setResult(finalResult);
      setState('dashboard');
    } catch (err: any) {
      console.error(err);
      const isTimeout = err.name === 'AbortError';
      const msg = isTimeout ? 'Request timed out after 10 minutes.' : err.message || 'Something went wrong.';
      setError(msg);
      setState('idle');
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 font-sans selection:bg-emerald-100 selection:text-emerald-900">
      <nav className="border-b border-zinc-200 bg-white/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div 
            className="flex items-center gap-2 cursor-pointer group" 
            onClick={() => { setState('idle'); setError(null); }}
          >
            <div className="w-8 h-8 bg-zinc-900 rounded-lg flex items-center justify-center group-hover:bg-emerald-600 transition-colors">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-xl tracking-tight">LectureAI</span>
          </div>
          
          {state === 'dashboard' && (
            <button 
              onClick={() => { setState('idle'); setError(null); }}
              className="text-sm font-semibold text-zinc-500 hover:text-zinc-900 transition-colors"
            >
              New Lecture
            </button>
          )}
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12">
        {error && state === 'idle' && (
          <div className="mb-6 max-w-2xl mx-auto p-4 bg-red-50 text-red-600 border border-red-200 rounded-xl">
            <strong>Error:</strong> {error}
          </div>
        )}
        {state === 'idle' && <UploadForm onUpload={handleUpload} />}
        {state === 'processing' && <ProcessingView />}
        {state === 'dashboard' && result && <Dashboard result={result} />}
      </main>

      <footer className="max-w-7xl mx-auto px-6 py-12 border-t border-zinc-200 text-center">
        <p className="text-sm text-zinc-400">
          Built with AI for the future of learning.
        </p>
      </footer>
    </div>
  );
}
