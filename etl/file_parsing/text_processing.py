import re
from nltk import sent_tokenize

def _replace_unicode_newlines(s: str):
    """replace unicode characters for newlines \x0b"""
    s = s.replace("\x0b", "\n")
    s = s.encode("ascii", "ignore").decode()
    return s


def _remove_extra_spaces(s: str):
    """Replace the over spaces"""
    s = re.sub("\n{2,}", "\n", s)
    s = re.sub("\s{2,}", " ", s)
    return s


def _fix_amyloid(s: str):
    s = re.sub("(AMYLOIDOSIS|amyloidosis|AMYLOID|amyloid)([A-Za-z]\.)", r"\1 . \2", s)
    s = s.replace("AMYOIDOSIS", "AMYLOIDOSIS")
    s = s.replace("amyoidosis", "amyloidosis")
    return s


def _fix_clinical(s: str):
    s = re.sub("([A-Za-z]+)(Clinical|CLINICAL)", r"\1. \2", s)
    return s


def _fix_consecutive_punct(s: str):
    """add space between :- or .?"""
    s = re.sub("([?\.\:\-])([?\.\:\-])", r"\1 \2", s)
    return s


def _fix_colons(s: str):
    """replace :- by : """
    s = re.sub("(:[\.\-])", r": ", s)
    return s


def _fix_periods(s: str):
    s = re.sub("(\.)([A-Z]\.\s)", r"\1 \2", s)
    s = re.sub("(\.)([A-Z]{2,}|[A-Z][a-z]{2,})", r"\1 \2", s)
    s = re.sub("(\.{2,})", r".", s)
    # s = re.sub('(\s\.)', r'.', s)
    return s


def _fix_double_dashes(s: str):
    """replace '--' with '\n' because they separate sections"""
    s = re.sub("(-{2,})", r"\n", s)
    return s


def _fix_question_marks(s: str):
    """replace .? by . """
    s = re.sub("(\.\?)", r". ", s)
    """replace '\n?' and '?\n' with '\n' because they separate sections"""
    s = re.sub("(\?)(?=\w)", r"\n", s)
    s = re.sub("\?\n|\n\?", r"\n", s)
    return s


def _newlines_to_periods(s: str):
    """replace '\n' with '. ' to tokenize sentences"""
    s = re.sub("\n", r". ", s)
    return s


def _fix_punctuation(s: str):
    """clean up punctuation"""
    # Pad punctuation
    s = re.sub("([\.,?!:\*])([A-Za-z]{2,})", r" \1", s)
    s = re.sub("(\w+)(\?)", r"\1 \2", s)
    return s


def _split_camel_case(s: str):
    """Some siginificant words were stuck together in camelCASE"""
    words = [[s[0]]]
    for c in s[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)
    return " ".join(["".join(word) for word in words])


def clean_cardiac_path(s: str):
    s = _replace_unicode_newlines(s)
    s = _fix_double_dashes(s)
    s = _fix_question_marks(s)
    s = _fix_clinical(s)
    s = _fix_amyloid(s)
    s = _remove_extra_spaces(s)
    s = _newlines_to_periods(s)
    s = _fix_periods(s)
    s = _fix_colons(s)
    s = _remove_extra_spaces(s)
    s = " ".join(sent_tokenize(s))
    return s.strip()


def clean_pyp(s: str):
    s = _remove_extra_spaces(s)
    return s


def extract_dates(s: str):
    # maybe convert to datetime, keep max
    # Match m/d/yy and mm/dd/yyyy, allowing any combination of one or two digits for the day and month, and two or four digits for the year
    # https://www.oreilly.com/library/view/regular-expressions-cookbook/9781449327453/ch04s04.html
    dates = re.findall(
        r"^(1[0-2]|0?[1-9])/(3[01]|[12][0-9]|0?[1-9])/(?:[0-9]{2})?[0-9]{2}$", s
    )
    return dates
