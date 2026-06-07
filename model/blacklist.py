import re
from typing import List

from rapidfuzz import fuzz

# Fuzzy-match acceptance thresholds. Fuzzy matching only helps for longer words
# where a Whisper mis-spelling is unambiguous; on short words a 1-character
# difference (atay≈patay, puto≈putot) is an innocent look-alike, not a typo, so
# we require exact whole-word matching below FUZZY_MIN_LEN.
FUZZY_MIN_LEN          = 6    # single words shorter than this are exact-match only
FUZZY_THRESHOLD_WORD   = 86
FUZZY_THRESHOLD_PHRASE = 88

# Compiled word-boundary patterns, built lazily and cached per term.
_PATTERNS: dict = {}


def _boundary_pattern(term: str) -> "re.Pattern":
    """Whole-word/phrase matcher. `term` is assumed already cleaned (lowercase,
    no punctuation, single-spaced). Spaces become flexible whitespace so phrases
    still match across normalized gaps."""
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)")


def _exact_match(term: str, text: str) -> bool:
    pat = _PATTERNS.get(term)
    if pat is None:
        pat = _boundary_pattern(term)
        _PATTERNS[term] = pat
    return pat.search(text) is not None


def word_matches(phrase: str, text: str) -> bool:
    """Whole-word / phrase match (public API). Single words match on word
    boundaries so ordinary words no longer trip a trigger by substring
    ("patay" no longer matches "atay"); multi-word phrases match with
    boundaries at the ends. Backed by the cached compiled patterns above."""
    return _exact_match(phrase, text)


def _fuzzy_match(term: str, tokens: List[str]) -> bool:
    parts = term.split()
    n = len(parts)
    if n == 1:
        if len(term) < FUZZY_MIN_LEN:
            return False  # too short for fuzzy — exact match already handled it
        return any(fuzz.ratio(term, tok) >= FUZZY_THRESHOLD_WORD for tok in tokens)
    # Multi-word phrase: slide an n-gram window of the same length over tokens.
    for i in range(len(tokens) - n + 1):
        window = " ".join(tokens[i:i + n])
        if fuzz.ratio(term, window) >= FUZZY_THRESHOLD_PHRASE:
            return True
    return False


def _find_hits(terms, text: str, tokens: List[str], fuzzy: bool = True) -> List[str]:
    """Whole-word exact match first; optional fuzzy fallback for ASR mis-spellings.
    Laughter markers pass fuzzy=False so a near-miss never wrongly suppresses an
    alert."""
    hits = []
    for term in terms:
        if _exact_match(term, text):
            hits.append(term)
        elif fuzzy and _fuzzy_match(term, tokens):
            hits.append(term)
    return hits

# ══════════════════════════════════════════════════════════════
# ECHOSENSE PRODUCTION BLACKLIST
# Davao City Grade 6 Classroom Bullying Detection
# Languages: Bisaya/Cebuano + Tagalog + English + Tagbis
# Categories: Academic, Appearance/Face, Body, Emotional, Threat
# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# HARD TRIGGERS
# These alone are enough to flag — severe, unambiguous words
# ─────────────────────────────────────────────────────────────
HARD_TRIGGERS = {

    # ── Bisaya Profanity ─────────────────────────────────────
    "yawa", "giatay", "bilat", "kayat", "iyot",
    "pesteng yawa", "bata og yawa", "piste", "punyeta",
    "hudas", "puta",
    # FIX 4 — removed "atay" (substring of common word "patay"/means 'liver')
    # and "buyag" (too common) — both tripped on normal speech.

    # ── Tagalog Profanity ────────────────────────────────────
    "putangina", "putang ina", "tang ina", "tangina",
    "pakyu", "anak ng puta", "anak og puta",
    "ampota", "pakshet", "hinayupak",

    # ── Threats ──────────────────────────────────────────────
    "patyon tika", "patyon ka nako", "kill you",
    "gusto kag sumbagay", "suwayi rag duol",
    "sumbagay ta", "papatayin kita",
    "papatayin kita", "mamamatay ka",

    # ── Severe Academic/Intelligence — Bisaya ────────────────
    "bogo", "bugok", "bulok", "way utok",
    "utok bolinaw", "utok munggos", "guba og utok",
    "sira og ulo", "kuwang og turnilyo",
    "monggi", "bogo kaayo", "bulok man ka",
    "bogo kaayo ka", "way utok ka",
    "way sulod ang utok", "bugits",

    # ── Severe Academic/Intelligence — Tagalog/English ───────
    "bobo", "tanga", "gago", "gaga", "ulol",
    "gunggong", "engot", "inutil", "luko luko",
    "retard", "sped", "monggoloid", "kupal",
    "sinto sinto",

    # ── Severe Appearance — Face Features ────────────────────
    # Bisaya face bullying
    "bungi",                    # cleft lip — very cruel
    "uling",                    # charcoal (dark skin)
    "murag uling",              # looks like charcoal
    "agta",                     # offensive dark skin comparison
    "murag ungo",               # looks like a monster
    "murag wakwak",             # looks like a witch
    "kulisap ang buhok",        # lice in hair
    "kuto",                     # lice
    "dakog ulo walay laman",    # big head nothing inside
    "nawong mo murag ungo",     # face like a monster
    "itom kaayo murag uling",   # so dark like charcoal
    "pango kaayo murag baboy",  # flat nosed like a pig
    "ngipon mo murag mais",     # teeth like corn
    "nawong mo murag piso",     # face flat like a coin
    "pangit ang nawong ug baho pa", # ugly and smelly
    "dakog ilong murag patatas", # big nose like potato
    "murag litson",             # looks like lechon (fat insult)
    "tambokikoy ka murag litson", # fat like roasted pig
    "uling ka kaayo",           # you are charcoal dark

    # Tagalog face bullying
    "negra", "negro",           # offensive skin color
    "nasunog",                  # burned (dark skin)
    "itim na itim",             # very dark
    "katulad ng uling",         # like charcoal
    "ang pangit ng mukha mo",   # your face is so ugly
    "ang itim mo parang uling", # dark like charcoal
    "mukha kang unggoy",        # face like a monkey
    "mukha kang baboy",         # face like a pig

    # ── Severe Body Shaming ───────────────────────────────────
    "tambokikoy",               # mockingly fat
    "agta",                     # dark skin comparison
    "bilatog iyak",             # vulgar anatomical insult
    "bilat sa imong iro",       # vulgar anatomical insult
    "isuka ka sa imuhang mama", # your mom regrets you
    "walay gustong makig uban sa imo", # nobody wants you

    # ── Severe Social ─────────────────────────────────────────
    "walang kwenta", "walang silbi",
    "wala kang kwenta", "hampaslupa",
    "nobody likes you", "nobody wants you",
    "you are nothing", "worthless", "go die",
    "wala kang kwenta pare",
    "mas maayo pa kung wala ka",
    "kolera", "anak og pobre",

    # ── ASR variant spellings (Whisper mishears these short Bisaya words) ──
    # Explicit aliases for hard triggers, since fuzzy matching is unsafe on
    # short words: bugo→bogo, bolok→bulok, bugog→bugok.
    "bugo", "bolok", "bugog",
}

# ─────────────────────────────────────────────────────────────
# SOFT TRIGGERS
# These need 2+ together OR repetition OR angry tone to flag
# ─────────────────────────────────────────────────────────────
SOFT_TRIGGERS = {

    # ── Mild exclamations / casual — not standalone bullying ──
    # Reclassified out of HARD_TRIGGERS: everyday expletives and playful Bisaya
    # ("away ta") that are not directed bullying on their own. Only count when
    # paired with another soft word or repeated.
    "leche", "lintik", "bwisit", "away ta",

    # 'pangit' (ugly) is casual-common ("pangit ang panahon"), so it is SOFT —
    # it needs a pair or repetition to count. 'pangid' is the common Whisper
    # mishearing of 'pangit'.
    "pangit", "pangid",

    # ── Mild Academic ─────────────────────────────────────────
    "slow kaayo", "hinay og pick up", "kuwang", "abno",
    "abnormal", "pabigat", "pabigat sa grupo",
    "kuwang kuwang", "kulanging",
    "moron",
    # FIX 4 — removed "olo" (too short/common), "stupid"/"idiot"/"dumb"
    # (everyday English in songs/movies/casual talk). Also removed "bobo"/"tanga"
    # here — they are HARD triggers; listing them in both sets made a single
    # "bobo" count as hard+soft and wrongly satisfy the hard+soft rule.

    # ── Face Features — Bisaya (soft) ────────────────────────
    "dakog ilong",              # big nose
    "dakog dunggan",            # big ears
    "dakog mata",               # big eyes
    "dakog ulo",                # big head
    "pango",                    # flat nose
    "nipis ang ngabil",         # thin lips
    "baga ang ngabil",          # thick lips
    "ngipon mo murag pader",    # teeth like wall
    "ngil-ad ang ngipon",       # ugly teeth
    "gahong ang nawong",        # pockmarked face
    "puno og pimple",           # full of pimples
    "mapula ang nawong",        # blotchy red face
    "mansa ang balat",          # blotchy skin
    "baho og kilikili",         # smelly armpits
    "baho og tiil",             # smelly feet
    "baho og baba",             # bad breath
    "baho og ilong",            # smelly nose
    "lugaon ang dunggan",       # dirty waxy ears
    "bulingon",                 # dirty/unwashed
    "gididit",                  # dirty
    "kaguron",                  # covered in scabs
    "mala ang buhok",           # dirty hair
    "mata mo murag baka",       # eyes like cow
    "mata mo murag isda",       # eyes like fish
    "libat",                    # cross-eyed
    "duling",                   # cross-eyed
    "upaw",                     # bald/bad haircut
    "pisot",                    # uncircumcised (humiliating)
    "giraf",                    # long neck/awkward tall
    "ilong mo murag patatas",   # nose like potato
    "ilong mo murag saging",    # nose like banana
    "baboy ang ilong",          # pig nose
    "ulo mo murag bayong",      # head like a basket
    "ulo mo murag bola",        # head like a ball
    "piso ang nawong",          # flat face like coin
    "himbis ang nawong",        # very flat face
    "ngilit ang mata",          # squinty eyes
    "dunggan mo murag elepante", # ears like elephant
    "baho og panit",            # smelly skin
    "utal",                     # stutter
    "utal ka man",              # you stutter

    # ── Face Features — Tagalog (soft) ───────────────────────
    "malaking ilong",           # big nose
    "malaking tenga",           # big ears
    "malaking mata",            # big eyes
    "pango",                    # flat nose (same word)
    "malaking ngipin",          # big teeth
    "baho ng hininga",          # bad breath
    "malalaking ngipin",        # big teeth
    "puro pimple",              # full of pimples
    "amoy pawis",               # smells like sweat
    "amoy araw",                # smells like sun (body odor)
    "hindi naliligo",           # doesn't bathe
    "madumi",                   # dirty
    "ilong mo parang patatas",  # nose like potato
    "baboy ang ilong mo",       # pig nose
    "mata mong parang baka",    # cow eyes
    "tenga mo parang elepante", # elephant ears
    "baho ng katawan",          # smelly body
    "daming pimple",            # lots of pimples
    "pangit ang mukha",         # ugly face
    "pangit na pangit",         # very ugly

    # ── Body Shape — Bisaya (soft) ───────────────────────────
    "tambok",                   # fat
    "itom kaayo",               # very dark
    # FIX 4 — removed standalone "baboy" (casual Filipino word) and "itom"
    # (matches normal "dark" descriptions). Specific phrases below still match.
    "putot",                    # short
    "pandak",                   # short
    "niwang",                   # skinny
    "niwang kaayo",             # very skinny
    "butong",                   # all bones
    "murag palito",             # like a matchstick
    "murag sundang",            # like a machete (thin)
    "dako og tiyan",            # big belly
    "tiyan mo murag buntis",    # belly like pregnant
    "dakog bulan",              # big moon belly
    "murag dwende",             # like a dwarf
    "murag nuno",               # like a gnome
    "murag baka",               # looks like a cow
    "pangit kaayo",             # very ugly

    # ── Body Shape — Tagalog (soft) ──────────────────────────
    "mataba",                   # fat
    "payat na payat",           # very skinny
    "tumbong na buto",          # all bones
    "parang stick",             # like a stick
    "parang buntis",            # looks pregnant
    "pandak",                   # short
    "parang dwarf",             # like a dwarf
    "smelly", "stinky",         # FIX 4 — removed casual English "ugly"/"fat"

    # ── Emotional Taunting ────────────────────────────────────
    "iiyak na yan",             # about to cry
    "hilak nasad",              # crying again
    "hilak hilak",              # crybaby sounds
    "hilakon",                  # always crying
    "sgeg hilak",               # always crying
    "pikon man diay",           # sore loser
    "oa kaayo",                 # overacting
    "bida bida",                # attention seeker
    "pabida",                   # attention seeker
    "cringe kaayo ka",          # very cringe
    "jejemon",                  # social exile
    "luod kaayo ka",            # you are disgusting
    "ampon",                    # adopted (used cruelly)
    "crybaby", "loser", "freak", "weirdo",

    # ── Social Exclusion ──────────────────────────────────────
    "pikon",                    # easily offended
    "sumbong",                  # tattletale
    "sumbongera",               # female tattletale
    "sumbongero",               # male tattletale
    "isugbo sa maam",           # go tell the teacher
    "sumbong sa maam",          # tell the teacher
    "uli sa inyo oy",           # go home
    "dili ka among friend",     # not our friend
    "exclude na ta siya",       # let's exclude them
    "ayaw siya apila",          # don't include them
    "wala kang amigo",          # you have no friends
    "ikaw ang problema",        # you are the problem
    "ayaw pagpakita diri",      # don't show your face here
    "gidat ugan",               # social outcast
    "wala kay labot",           # you don't belong
    "isugbo nako ka",           # i will report you
    "sumbong didto sa imong mama", # go cry to your mom
    "loser", "get lost", "go away",
    "nobody cares", "you don t belong",
    "dili ka among barkada",    # not part of our group

    # ── Code-switch combo phrases ─────────────────────────────
    "bogo ka man gyud",         # you really are stupid
    "pangit kaayo imong nawong", # your face is very ugly
    "tambok kaayo ka",          # you are very fat
    "itom kaayo ka",            # you are very dark
}

# ─────────────────────────────────────────────────────────────
# LAUGHTER / CASUAL MARKERS
# If these appear with soft triggers = kantiyawan, suppress alert
# Exception: hard trigger + angry tone overrides even laughter
# ─────────────────────────────────────────────────────────────
LAUGHTER_MARKERS = {
    "haha", "hehe", "hihi", "ahaha", "ahahaha",
    "lol", "char", "charot", "joke", "biro",
    "joke lang", "char lang", "naa bay", "sus",
    "grabe", "hala", "peace", "peace out",
    "cge lang", "wala lang", "biro ra",
}

# ─────────────────────────────────────────────────────────────
# SEVERITY MAPPING
# ─────────────────────────────────────────────────────────────
HIGH_SEVERITY_WORDS = {
    "yawa", "giatay", "bilat", "kayat", "iyot",
    "putangina", "putang ina", "tangina", "pakyu",
    "patyon tika", "patyon ka nako", "kill you",
    "puta", "anak ng puta", "pesteng yawa",
    "gusto kag sumbagay", "monggi", "retard",
    "papatayin kita", "go die", "uling", "murag uling",
    "agta", "negra", "negro", "bungi",
    "kulisap ang buhok", "mukha kang unggoy",
}

MEDIUM_SEVERITY_WORDS = {
    "bogo", "bugok", "bulok", "gago", "bobo",
    "tanga", "ulol", "way utok", "walang kwenta",
    "wala kang kwenta", "walay gustong makig uban sa imo",
    "isuka ka sa imuhang mama", "worthless", "inutil",
    "bugits", "bogo kaayo", "bulok man ka",
    "tambokikoy", "murag litson", "murag ungo",
    "nawong mo murag ungo", "itom kaayo murag uling",
    "dakog ulo walay laman", "ang pangit ng mukha mo",
}

LOW_SEVERITY_WORDS = {
    "pangit", "tambok", "itom", "putot", "baho",
    "pikon", "sumbong", "hilak nasad", "iiyak na yan",
    "ugly", "fat", "crybaby", "loser", "freak",
    "ampon", "luod kaayo ka", "dakog ilong",
    "dakog dunggan", "pango", "niwang", "pandak",
    "malaking ilong", "malaking tenga",
}

# ─────────────────────────────────────────────────────────────
# CATEGORY KEYWORD SETS (for classification)
# ─────────────────────────────────────────────────────────────
ACADEMIC_KEYWORDS = {
    "bogo", "bugok", "bulok", "bobo", "tanga", "way utok",
    "retard", "sped", "slow kaayo", "guba og utok", "inutil",
    "gunggong", "stupid", "idiot", "dumb", "moron", "ulol",
    "kuwang", "abno", "abnormal", "pabigat", "sinto sinto",
    "utok bolinaw", "utok munggos", "monggi", "kupal",
}

APPEARANCE_KEYWORDS = {
    "dakog ilong", "pango", "bungi", "uling", "agta",
    "dakog dunggan", "dakog mata", "dakog ulo", "murag",
    "pangit ang nawong", "itim na itim", "nasunog",
    "malaking ilong", "kulisap", "kuto", "utal", "duling",
    "libat", "upaw", "pisot", "giraf", "lugaon",
    "baho og kilikili", "baho og tiil", "baho og baba",
    "baho ng hininga", "amoy pawis", "amoy araw",
    "puno og pimple", "gahong ang nawong", "mala ang buhok",
    "ilong mo murag", "mata mo murag", "ulo mo murag",
    "nawong mo murag", "ngipon mo murag", "negra", "negro",
}

BODY_KEYWORDS = {
    "tambok", "tambokikoy", "baboy", "putot", "pandak",
    "niwang", "butong", "murag palito", "murag sundang",
    "dako og tiyan", "murag dwende", "murag nuno",
    "murag baka", "murag litson", "mataba", "payat",
    "parang stick", "parang buntis",
}

EMOTIONAL_KEYWORDS = {
    "hilak nasad", "iiyak na yan", "pikon", "ampon",
    "wala kang kwenta", "walay gustong", "crybaby", "loser",
    "hilakon", "sgeg hilak", "luod kaayo ka", "cringe",
    "bida bida", "oa kaayo", "gidat ugan",
}

THREAT_KEYWORDS = {
    "patyon", "kill", "sumbagay", "away ta",
    "suwayi", "papatayin", "mamamatay",
}

# ─────────────────────────────────────────────────────────────
# COMBINED SET
# ─────────────────────────────────────────────────────────────
ALL_BLACKLIST = HARD_TRIGGERS | SOFT_TRIGGERS


# ─────────────────────────────────────────────────────────────
# PHONETIC VARIANTS
# Whisper-tiny mishears specific Bisaya words in *predictable* ways
# (young high-pitched voices + int8 model). Rather than loosen the
# fuzzy threshold globally (unsafe on short words), we rewrite these
# known mishearings to the canonical blacklist word BEFORE matching.
# Keys are what Whisper *writes*; values are the real word.
# ─────────────────────────────────────────────────────────────
PHONETIC_VARIANTS = {
    # Whisper hears → actual blacklist word
    "bubo":     "bobo",
    "boba":     "bobo",
    "bubo ka":  "bobo",
    "bugo":     "bogo",
    "buga":     "bogo",
    "bugog":    "bugok",
    "bolok":    "bulok",
    "buluk":    "bulok",
    "boluk":    "bulok",
    "pang":     "pangit",
    "pangid":   "pangit",
    "panget":   "pangit",
    "tambok":   "tambok",
    "tambuk":   "tambok",
    "gagu":     "gago",
    "gagu ka":  "gago",
    "yava":     "yawa",
    "yaba":     "yawa",
    "iawa":     "yawa",
    "tanga":    "tanga",
    "tanka":    "tanga",
    "bobu":     "bobo",
    "bubu":     "bobo",
    "bugu":     "bogo",
    "hilak":    "hilak nasad",
    "hila":     "hilak nasad",
    "pikon":    "pikon",
    "pican":    "pikon",
    "ampon":    "ampon",
    "hampon":   "ampon",
    "dakog":    "dakog ilong",
    "dakug":    "dakog ilong",
    "pango":    "pango",
    "panga":    "pango",
    "uling":    "uling",
    "ulin":     "uling",
    "bungi":    "bungi",
    "bunge":    "bungi",

    # ── Field-observed Grade 6 mishearings (added) ─────────────────────────
    # Only genuinely NEW keys below; entries already mapped above (bugo, pangid,
    # panget, pang, tambuk, gagu, yava/yaba/iawa, hilak, dakog, pango, ulin,
    # bunge, …) and pointless single-word self-maps are intentionally omitted.

    # dakog ilong
    "daku bilong":  "dakog ilong",
    "dacoge illan": "dakog ilong",
    "dakog bilong": "dakog ilong",
    "dako ilong":   "dakog ilong",
    "dakong ilong": "dakog ilong",
    "dakog along":  "dakog ilong",
    "daku along":   "dakog ilong",
    # dakog mata
    "dakog matha":  "dakog mata",
    "daku mata":    "dakog mata",
    "dacoge mata":  "dakog mata",
    "dakog mato":   "dakog mata",
    # pangit
    "bangit":       "pangit",
    "banggit":      "pangit",
    # dakog dunggan
    "dakog dungan": "dakog dunggan",
    "daku dunggan": "dakog dunggan",
    "dakog dongan": "dakog dunggan",
    # tambok
    "tambog":       "tambok",
    "thambok":      "tambok",
    # pango
    "panggo":       "pango",
    "pango ka":     "pango",
    # bungi
    "bongi":        "bungi",
    # uling
    "ooling":       "uling",
    # hilak nasad — keep the 2-word form intact (avoids "hilak nasad nasad")
    "hilak nasad":  "hilak nasad",
    "hila nasad":   "hilak nasad",
    # bogo
    "boga":         "bogo",
    "bogoh":        "bogo",
    # yawa
    "yowa":         "yawa",
    # niwang
    "niwangan":     "niwang",
    "niwag":        "niwang",
}


def apply_phonetic_variants(text: str) -> str:
    """Rewrite known Whisper mishearings to their canonical blacklist word.
    Greedy: tries a two-word phrase match (e.g. 'bubo ka') before falling back
    to single-word substitution. Operates on already-cleaned lowercase text."""
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        # Try two-word match first
        if i + 1 < len(words):
            two_word = words[i] + ' ' + words[i + 1]
            if two_word in PHONETIC_VARIANTS:
                result.append(PHONETIC_VARIANTS[two_word])
                i += 2
                continue
        # Try single word match
        if words[i] in PHONETIC_VARIANTS:
            result.append(PHONETIC_VARIANTS[words[i]])
        else:
            result.append(words[i])
        i += 1
    corrected = ' '.join(result)
    if corrected != text:
        print(f"[VARIANTS] '{text}' → '{corrected}'")
    return corrected


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_word_severity(word: str) -> str:
    if word in HIGH_SEVERITY_WORDS:
        return "high"
    if word in MEDIUM_SEVERITY_WORDS:
        return "medium"
    return "low"


def check_transcript(transcript: str) -> dict:
    text = clean_text(transcript)
    # Correct predictable Whisper mishearings BEFORE any matching, so canonical
    # blacklist words are what we test against (see PHONETIC_VARIANTS).
    text = apply_phonetic_variants(text)
    tokens = text.split()

    # Whole-word matching (+ fuzzy fallback for ASR mis-spellings) so ordinary
    # words no longer trip a trigger by substring (e.g. "patay" no longer hits
    # "atay"). Laughter uses exact-only to avoid wrongly suppressing alerts.
    hard_hits = _find_hits(HARD_TRIGGERS, text, tokens)
    soft_hits = _find_hits(SOFT_TRIGGERS, text, tokens)
    laughing  = _find_hits(LAUGHTER_MARKERS, text, tokens, fuzzy=False)

    # Safety: a term must never count as BOTH hard and soft (e.g. a word listed
    # in both sets), otherwise one hard word alone would trip the hard+soft rule.
    soft_hits = [w for w in soft_hits if w not in HARD_TRIGGERS]

    has_hard      = len(hard_hits) > 0
    has_soft_pair = len(soft_hits) >= 2
    is_casual     = len(laughing) > 0
    has_profanity = has_hard or has_soft_pair

    all_detected = list(set(hard_hits + soft_hits))

    # Severity
    severity = "low"
    for w in all_detected:
        if w in HIGH_SEVERITY_WORDS:
            severity = "high"
            break
        elif w in MEDIUM_SEVERITY_WORDS and severity != "high":
            severity = "medium"

    # Categories
    categories = []
    for w in all_detected:
        if any(a in w for a in ACADEMIC_KEYWORDS):
            categories.append("academic_shaming")
        if any(a in w for a in APPEARANCE_KEYWORDS):
            categories.append("appearance_shaming")
        if any(b in w for b in BODY_KEYWORDS):
            categories.append("body_shaming")
        if any(e in w for e in EMOTIONAL_KEYWORDS):
            categories.append("emotional_taunting")
        if any(t in w for t in THREAT_KEYWORDS):
            categories.append("threat")

    return {
        "has_profanity":  has_profanity,
        "detected_words": all_detected,
        "hard_hits":      hard_hits,
        "soft_hits":      soft_hits,
        "is_casual":      is_casual,
        "severity":       severity,
        "categories":     list(set(categories)),
        "word_count":     len(all_detected),
        "checked_text":   text,   # post-variant text actually matched (for [CHECK] log)
    }


# --- Backward-compat shims --------------------------------------------------
# The legacy Vosk path (model/vosk_detect.py) imports these. The live pipeline
# now uses check_transcript() via model/whisper_stt.py, but these keep the old
# module importable so nothing breaks if it is loaded.
def contains_blacklisted_word(text: str) -> bool:
    text_lower = text.lower()
    return any(word in text_lower for word in ALL_BLACKLIST)


def get_detected_words(text: str) -> List[str]:
    text_lower = text.lower()
    return [word for word in ALL_BLACKLIST if word in text_lower]
