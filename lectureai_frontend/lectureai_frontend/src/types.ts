export interface SlideNote {
  slideNumber: number;
  title: string;
  content: string;
}

export interface Flashcard {
  id: string;
  question: string;
  answer: string;
}

export interface QuestionAnswer {
  id: string;
  question: string;
  answer: string;
}

export interface LectureResult {
  id: string;
  title: string;
  notes: SlideNote[];
  summary: string;
  flashcards: Flashcard[];
  questions: QuestionAnswer[];
}

export type AppState = 'idle' | 'processing' | 'dashboard';
