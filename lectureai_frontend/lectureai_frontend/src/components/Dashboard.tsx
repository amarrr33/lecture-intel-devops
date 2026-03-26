import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { FileText, BookOpen, Layers, HelpCircle, ChevronRight, ChevronDown, Download, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { LectureResult } from '../types';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

interface DashboardProps {
  result: LectureResult;
}

export function Dashboard({ result }: DashboardProps) {
  const [activeTab, setActiveTab] = useState<'notes' | 'summary' | 'flashcards' | 'qa'>('notes');
  const [isExporting, setIsExporting] = useState(false);
  const exportRef = useRef<HTMLDivElement>(null);

  const tabs = [
    { id: 'notes', label: 'Notes', icon: FileText },
    { id: 'summary', label: 'Summary', icon: BookOpen },
    { id: 'flashcards', label: 'Flashcards', icon: Layers },
    { id: 'qa', label: 'Q&A', icon: HelpCircle },
  ] as const;

  const handleExportPDF = async () => {
    if (!exportRef.current) return;
    setIsExporting(true);

    try {
      const canvas = await html2canvas(exportRef.current, {
        scale: 2,
        useCORS: true,
        logging: false,
        backgroundColor: '#ffffff'
      });

      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'px',
        format: [canvas.width, canvas.height]
      });

      pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
      pdf.save(`${result.title.toLowerCase().replace(/\s+/g, '_')}_study_material.pdf`);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto">
      <div className="mb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-4xl font-bold text-zinc-900 mb-2">{result.title}</h1>
          <p className="text-zinc-500">Generated study materials for your lecture.</p>
        </div>
        <button
          onClick={handleExportPDF}
          disabled={isExporting}
          className="flex items-center gap-2 px-6 py-3 bg-emerald-600 text-white font-bold rounded-xl hover:bg-emerald-700 transition-all shadow-lg shadow-emerald-600/10 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isExporting ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Exporting...
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              Export PDF
            </>
          )}
        </button>
      </div>

      {/* Tabs */}
      <div className="flex p-1 bg-zinc-100 rounded-2xl mb-8 w-fit">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${
              activeTab === tab.id
                ? 'bg-white text-zinc-900 shadow-sm'
                : 'text-zinc-500 hover:text-zinc-700'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="min-h-[600px]">
        <AnimatePresence mode="wait">
          {activeTab === 'notes' && (
            <motion.div
              key="notes"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="grid gap-6"
            >
              {result.notes.map((note) => (
                <div key={note.slideNumber} className="bg-white p-8 rounded-3xl border border-zinc-100 shadow-sm">
                  <div className="flex items-center gap-4 mb-4">
                    <span className="px-3 py-1 bg-zinc-900 text-white text-xs font-bold rounded-full">
                      SLIDE {note.slideNumber}
                    </span>
                    <h3 className="text-xl font-bold text-zinc-900">{note.title}</h3>
                  </div>
                  <div className="prose prose-zinc max-w-none text-zinc-600 leading-relaxed">
                    <ReactMarkdown>{note.content}</ReactMarkdown>
                  </div>
                </div>
              ))}
            </motion.div>
          )}

          {activeTab === 'summary' && (
            <motion.div
              key="summary"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="bg-white p-10 rounded-3xl border border-zinc-100 shadow-sm prose prose-zinc max-w-none"
            >
              <ReactMarkdown>{result.summary}</ReactMarkdown>
            </motion.div>
          )}

          {activeTab === 'flashcards' && (
            <motion.div
              key="flashcards"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="grid grid-cols-1 md:grid-cols-2 gap-6"
            >
              {result.flashcards.map((card) => (
                <FlashcardItem key={card.id} card={card} />
              ))}
            </motion.div>
          )}

          {activeTab === 'qa' && (
            <motion.div
              key="qa"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="space-y-4"
            >
              {result.questions.map((qa) => (
                <QAItem key={qa.id} qa={qa} />
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Hidden Export Template */}
      <div className="fixed -left-[9999px] top-0">
        <div ref={exportRef} style={{ backgroundColor: '#ffffff', color: '#18181b' }} className="w-[800px] p-12 font-sans">
          <div style={{ borderBottom: '2px solid #18181b' }} className="pb-8 mb-12">
            <h1 style={{ color: '#18181b' }} className="text-4xl font-bold mb-4">{result.title}</h1>
            <p style={{ color: '#71717a' }}>Lecture Study Materials • Generated by LectureAI</p>
          </div>

          <section className="mb-16">
            <h2 style={{ color: '#18181b' }} className="text-2xl font-bold mb-8 flex items-center gap-3">
              <FileText className="w-6 h-6" style={{ color: '#18181b' }} /> Lecture Notes
            </h2>
            <div className="space-y-8">
              {result.notes.map((note) => (
                <div key={note.slideNumber} style={{ borderLeft: '4px solid #f4f4f5' }} className="pl-6">
                  <div className="flex items-center gap-3 mb-2">
                    <span style={{ backgroundColor: '#18181b', color: '#ffffff' }} className="text-xs font-bold px-2 py-0.5 rounded">Slide {note.slideNumber}</span>
                    <h3 style={{ color: '#18181b' }} className="font-bold text-xl">{note.title}</h3>
                  </div>
                  <p style={{ color: '#52525b' }} className="leading-relaxed">{note.content}</p>
                </div>
              ))}
            </div>
          </section>

          <section className="mb-16">
            <h2 style={{ color: '#18181b' }} className="text-2xl font-bold mb-8 flex items-center gap-3">
              <BookOpen className="w-6 h-6" style={{ color: '#18181b' }} /> Summary
            </h2>
            <div style={{ color: '#3f3f46' }}>
              <ReactMarkdown
                components={{
                  h1: ({ children }) => <h1 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '16px', color: '#18181b' }}>{children}</h1>,
                  h2: ({ children }) => <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginTop: '24px', marginBottom: '12px', color: '#18181b' }}>{children}</h2>,
                  h3: ({ children }) => <h3 style={{ fontSize: '18px', fontWeight: 'bold', marginTop: '20px', marginBottom: '8px', color: '#18181b' }}>{children}</h3>,
                  p: ({ children }) => <p style={{ marginBottom: '12px', lineHeight: '1.6', color: '#3f3f46' }}>{children}</p>,
                  ul: ({ children }) => <ul style={{ listStyleType: 'disc', paddingLeft: '24px', marginBottom: '12px', color: '#3f3f46' }}>{children}</ul>,
                  li: ({ children }) => <li style={{ marginBottom: '4px', color: '#3f3f46' }}>{children}</li>,
                  blockquote: ({ children }) => (
                    <blockquote style={{ borderLeft: '4px solid #10b981', paddingLeft: '16px', fontStyle: 'italic', color: '#71717a', margin: '24px 0' }}>
                      {children}
                    </blockquote>
                  ),
                }}
              >
                {result.summary}
              </ReactMarkdown>
            </div>
          </section>

          <section className="mb-16">
            <h2 style={{ color: '#18181b' }} className="text-2xl font-bold mb-8 flex items-center gap-3">
              <Layers className="w-6 h-6" style={{ color: '#18181b' }} /> Flashcards
            </h2>
            <div className="grid grid-cols-2 gap-6">
              {result.flashcards.map((card) => (
                <div key={card.id} style={{ border: '1px solid #f4f4f5', backgroundColor: '#fafafa' }} className="p-6 rounded-2xl">
                  <p style={{ color: '#059669' }} className="text-xs font-bold uppercase mb-2">Q: {card.question}</p>
                  <p style={{ color: '#3f3f46' }} className="text-sm font-medium">A: {card.answer}</p>
                </div>
              ))}
            </div>
          </section>

          <section>
            <h2 style={{ color: '#18181b' }} className="text-2xl font-bold mb-8 flex items-center gap-3">
              <HelpCircle className="w-6 h-6" style={{ color: '#18181b' }} /> Q&A
            </h2>
            <div className="space-y-6">
              {result.questions.map((qa) => (
                <div key={qa.id} style={{ border: '1px solid #f4f4f5' }} className="p-6 rounded-2xl">
                  <p style={{ color: '#18181b' }} className="font-bold mb-2">{qa.question}</p>
                  <p style={{ color: '#52525b' }}>{qa.answer}</p>
                </div>
              ))}
            </div>
          </section>

          <div style={{ marginTop: '80px', paddingTop: '32px', borderTop: '1px solid #f4f4f5', color: '#a1a1aa' }} className="text-center text-xs">
            Generated with LectureAI • {new Date().toLocaleDateString()}
          </div>
        </div>
      </div>
    </div>
  );
}

function FlashcardItem({ card }: { card: { question: string; answer: string } }) {
  const [isFlipped, setIsFlipped] = useState(false);

  return (
    <div
      className="h-64 perspective-1000 cursor-pointer group"
      onClick={() => setIsFlipped(!isFlipped)}
    >
      <motion.div
        className="relative w-full h-full transition-all duration-500 preserve-3d"
        animate={{ rotateY: isFlipped ? 180 : 0 }}
      >
        {/* Front */}
        <div className="absolute inset-0 backface-hidden bg-white p-8 rounded-3xl border border-zinc-100 shadow-sm flex flex-col items-center justify-center text-center">
          <span className="text-xs font-bold text-emerald-600 uppercase tracking-widest mb-4">Question</span>
          <p className="text-lg font-semibold text-zinc-900">{card.question}</p>
          <div className="mt-8 text-zinc-400 text-xs flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            Click to flip <ChevronRight className="w-3 h-3" />
          </div>
        </div>

        {/* Back */}
        <div
          className="absolute inset-0 backface-hidden bg-zinc-900 p-8 rounded-3xl shadow-xl flex flex-col items-center justify-center text-center text-white"
          style={{ transform: 'rotateY(180deg)' }}
        >
          <span className="text-xs font-bold text-emerald-400 uppercase tracking-widest mb-4">Answer</span>
          <p className="text-lg leading-relaxed">{card.answer}</p>
        </div>
      </motion.div>
    </div>
  );
}

function QAItem({ qa }: { qa: { question: string; answer: string } }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="bg-white rounded-2xl border border-zinc-100 shadow-sm overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-8 py-6 flex items-center justify-between text-left hover:bg-zinc-50 transition-colors"
      >
        <span className="font-bold text-zinc-900">{qa.question}</span>
        {isOpen ? <ChevronDown className="w-5 h-5 text-zinc-400" /> : <ChevronRight className="w-5 h-5 text-zinc-400" />}
      </button>
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            className="overflow-hidden"
          >
            <div className="px-8 pb-8 text-zinc-600 leading-relaxed border-t border-zinc-50 pt-4">
              {qa.answer}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
