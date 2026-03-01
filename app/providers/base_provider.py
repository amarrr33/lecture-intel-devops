from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers supporting the lecture pipeline.
    """
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def used_internet(self) -> bool:
        pass

    @abstractmethod
    def generate_notes(self, lecture_text: str, speech_text: str = "", max_tokens: int = 400) -> str:
        """Generates study notes from lecture content and optional speech context."""
        pass

    @abstractmethod
    def generate_summary(self, lecture_text: str) -> str:
        """Generates a detailed summary of the entire lecture."""
        pass

    @abstractmethod
    def generate_flashcards(self, lecture_text: str, n: int = 20) -> List[Dict[str, str]]:
        """Generates flashcards for the lecture."""
        pass

    @abstractmethod
    def generate_questions_with_answers(self, lecture_text: str) -> Dict[str, List[Dict[str, str]]]:
        """Generates an exam pack containing sections of questions."""
        pass

    @abstractmethod
    def generate_subject_summary(self, subject_corpus: str, syllabus: str) -> str:
        """Generates a mega-summary for the entire subject corpus."""
        pass

    @abstractmethod
    def generate_subject_flashcards(self, subject_corpus: str, syllabus: str, n: int = 50) -> List[Dict[str, str]]:
        """Generates flashcards across the entire subject corpus."""
        pass

    @abstractmethod
    def generate_subject_questions_with_answers(self, subject_corpus: str, syllabus: str) -> Dict[str, List[Dict[str, str]]]:
        """Generates a massive exam pack across the entire subject corpus."""
        pass
