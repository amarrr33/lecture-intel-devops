import { Loader2, Sparkles, Brain, FileText, Zap } from 'lucide-react';
import { motion } from 'motion/react';

const steps = [
  { icon: FileText, label: 'Extracting slides...' },
  { icon: Zap, label: 'Transcribing audio...' },
  { icon: Brain, label: 'Aligning content...' },
  { icon: Sparkles, label: 'Generating materials...' },
];

export function ProcessingView() {
  return (
    <div className="max-w-md mx-auto text-center py-20">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        className="inline-block mb-8"
      >
        <Loader2 className="w-16 h-16 text-emerald-500" />
      </motion.div>
      
      <h2 className="text-2xl font-bold text-zinc-900 mb-4">Processing Lecture</h2>
      <p className="text-zinc-500 mb-12">Our AI is working its magic to transform your lecture into study materials.</p>

      <div className="space-y-6">
        {steps.map((step, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 1.5 }}
            className="flex items-center gap-4 bg-white p-4 rounded-2xl border border-zinc-100 shadow-sm"
          >
            <div className="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center">
              <step.icon className="w-5 h-5 text-emerald-600" />
            </div>
            <span className="font-medium text-zinc-700">{step.label}</span>
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: '100%' }}
              transition={{ delay: i * 1.5, duration: 1.5 }}
              className="ml-auto h-1 bg-emerald-100 rounded-full overflow-hidden"
            >
              <div className="h-full bg-emerald-500" />
            </motion.div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
