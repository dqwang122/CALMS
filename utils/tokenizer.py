

class SBDSplitor:
    def __init__(self, language):
        self.language = language
        try:
            from pysbd import Segmenter
            self._seg = Segmenter(language=language, clean=False)
        except ImportError:
            raise ImportError('Please install pySBD splitor with: pip install pysbd')
        except ValueError:
            self._seg = Segmenter(language='en', clean=False)
        self._seg.segment("warmup for pysbd splitor")

    def split(self, text):
        if self.language == 'en':
            from nltk import sent_tokenize
            return sent_tokenize(text)
        return self._seg.segment(text)


class MosesTokenizer:
    def __init__(self, language):
        self.language = language
        try:
            from sacremoses import MosesTokenizer
            self._tok = MosesTokenizer(lang=language)
        except ImportError:
            raise ImportError('Please install Moses tokenizer with: pip install sacremoses')
        self._tok.tokenize("warmup for mose tokenizer")

    def tokenize(self, text, return_str=True):
        return self._tok.tokenize(text, return_str=return_str)
