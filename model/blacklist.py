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

    # ── Grade 6 Bisaya (Davao) — field-observed, added ───────
    "buang",            # crazy/stupid — most-used Bisaya insult
    "buang ka",         # you are crazy
    "buang kaayo",      # very crazy
    "ulaga",            # idiot/fool in Bisaya
    "bungoan tika",     # I will punch your face (threat)
    "suntukan ta",      # let's fight (threat)
    "away ta",          # let's fight, Bisaya (promoted from SOFT_TRIGGERS)
    "sampalan tika",    # I will slap you (threat)

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

    # ── English bullying sentences (Grade 6) — added ─────────
    # Contractions are stored apostrophe-free with a space ("you re ...") because
    # clean_text() turns "you're" into "you re"; the "youre" -> "you re" phonetic
    # variant normalizes the no-apostrophe spelling.
    "you are so ugly", "you re so ugly",
    "you are stupid", "you re stupid",
    "you are dumb", "you re dumb",
    "you are an idiot", "you re an idiot",
    "you are fat", "you re fat",
    "you are ugly", "shut up",
    "nobody likes you", "no one likes you",
    "you are worthless", "you are useless",
    "go away loser", "you are a loser",
    "you are disgusting", "you smell bad",
    "you are poor", "you are dirty",
    "kill yourself", "go kill yourself",

    # ── Tagalog bullying sentences (Grade 6) — added ─────────
    "ikaw ay pangit", "pangit ka talaga",
    "ikaw ay bobo", "napaka bobo mo", "napaka tanga mo",
    "ang tanga mo", "ang bobo mo",
    "walang kwenta ka", "walang silbi ka", "hampaslupa ka",
    "walang kwenta ang buhay mo", "patay ka na sana",
    "napaka sama mo", "ikaw ay basura", "basura ka",
    "mukha kang basura", "ang dumi mo",
    "hindi ka namin kailangan", "ayaw namin sa iyo",
    "wala kang kaibigan", "walang nagmamahal sa iyo",
    "palayasin mo ang sarili mo",

    # ── Bisaya bullying sentences (Grade 6) — added ──────────
    "buang kaayo ka", "wala kay kapuse pahan", "wala kay pulos",
    "yawa ka gyud", "piste ka", "giatay ka", "bogo gyud ka",
    "walay pulos ka", "dili ka gusto sa tanan", "dili ka gusto",
    "wala kay bili",

    # ── Animal comparisons (very common Grade 6) — added ─────
    "murag baboy ka", "murag unggoy ka", "murag iro ka",
    "murag baka ka", "murag manok ka",
    "mukha kang baboy", "mukha kang unggoy", "mukha kang aso",
    "mukha kang hayop", "para kang hayop",
    "animal ka", "hayop ka",

    # ── Mocking / taunting phrases — added ───────────────────
    "nganong ingon ana imong nawong", "tan awa imong nawong",
    "tan awa ka sa salamin", "hala ka", "kadiyot nimo",
    "kataw anan ka", "katawa tawa ka", "kawatan ka",

    # ── Strong appearance insults — added ────────────────────
    "pangit kaayo imong nawong", "nawong mo pang kalye",
    "nawong nimo pang banyo", "nawong mo murag binuksan",
    "nawong mo murag guba", "basag ang nawong mo",
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
    "leche", "lintik", "bwisit",

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

    # ── Grade 6 Bisaya (Davao) — field-observed, added ───────
    "bungal",                   # missing tooth
    "bungal ka",                # you have a missing tooth
    "walay amigo",              # no friends (Bisaya)
    "wala kay amigo",           # you have no friends
    "wala kay barkada",         # you have no friends
    "walay barkada",            # no friends
    "taas kaayo ka",            # you are too tall
    "gamay kaayo ka",           # you are too small
    "pandak kaayo",             # very short
    "buto buto",                # very bony / skinny
    "kalansay",                 # skeleton — very thin
    "unano ka",                 # you are a dwarf
    "dilaw ang ngipon",         # yellow teeth
    "bulok ang ngipon",         # rotten teeth
    "wala kay ngipon",          # you have no teeth

    # ── English bullying phrases (Grade 6) — added ───────────
    # Casual English singles (ugly/stupid/dumb/idiot/...) were removed earlier as
    # false-alarm prone; re-added per request. Safe now because a lone soft word
    # only DETECTS — it cannot alert without a pair, repetition, or angry audio.
    "ugly", "so ugly", "very ugly", "you re ugly",
    "stupid", "so stupid", "very stupid",
    "dumb", "so dumb", "idiot",
    "fatso", "fatty", "fat ass",
    "skinny", "too skinny", "too fat", "too tall", "too short",
    "you smell", "dirty", "so dirty",
    "big loser", "such a loser", "loser nerd",
    "cry baby", "stop crying", "you always cry",
    "nobody wants you", "no one wants you",
    "you have no friends", "friendless", "you are alone",
    "ugly face", "big nose", "flat nose", "big ears",
    "bad teeth", "yellow teeth", "bucktooth", "four eyes",
    "nerd", "geek",

    # ── Tagalog bullying phrases (Grade 6) — added ───────────
    "ang pangit mo", "napaka pangit",
    "taba mo", "mataba ka", "payat na payat",
    "pandak ka", "maitim ka",
    "amoy ka", "dami mong pimple", "pimple face",
    "malaking ilong mo", "sarat ang ilong mo", "malaking tenga mo",
    "ngipin mo", "dilaw ang ngipin mo", "baluktot ang ngipin",
    "suplado", "suplada", "epal", "arte",
    "mayabang", "hambog", "plastik", "ksp",
    "feeling maganda", "feeling gwapo", "sosyal na sosyal",
    "baduy", "jologs", "luma", "outdated",
    "bata pa", "immature", "cry baby ka", "iyakin",
    "pokpok", "siraulo", "baliw", "loka loka", "ulyanin",

    # ── Bisaya bullying phrases (Grade 6) — added ────────────
    "baho ka", "baho imong baba", "baho imong kilikili",
    "baho imong tiil", "mala imong buhok", "libog imong buhok",
    "hugaw ka", "dili ka naligo",
    "puti kaayo murag multo", "gahong nawong", "himbis nawong",
    "dabok", "gahi og ulo", "bugo sa tanan", "wala kay ulinapon",
    "wala kay kwenta", "dili maayo ka", "bastos ka", "supak ka",
    "hambogero", "hambogera", "burikat", "tarantado",
    "grabi ka", "way batasan", "way ugali",

    # ── Weight bullying (most common in PH Grade 6) — added ───
    "tambok", "tambok ka", "tambok kaayo", "taba mo", "mataba ka",
    "ang taba mo", "ang tambok nimo", "tambok na tambok",
    "baboy ka", "baboy ang katawan", "murag baboy",
    "fat ka", "ang fat mo", "chubby", "ang chubby mo", "overweight ka",

    # ── Nose bullying (very common) — added ──────────────────
    "dakog ilong", "dakog ilong mo", "ilong mo murag patatas",
    "ilong mo murag saging", "ilong mo murag bola",
    "pango ka", "pango kaayo", "flat nose ka",
    "pango ang ilong mo", "baboy ang ilong mo", "sarat ang ilong",

    # ── Skin color bullying (very common PH) — added ─────────
    "itom ka", "itom kaayo ka", "uling ka", "murag uling",
    "negro ka", "negra ka", "nag sunog ka", "nasusunog",
    "burnt ka", "gwapa lang kung puti", "pangit kay itom",
    "maputi ka sana",

    # ── Face / eyes bullying — added ─────────────────────────
    "duling ka", "libat ka", "duling ang mata mo",
    "mata mo murag baka", "mata mo murag isda", "dakog mata mo",
    "maliit ang mata mo", "ngilit ang mata",

    # ── Height / size bullying — added ───────────────────────
    "pandak ka", "pandak kaayo", "ang pandak mo", "putot ka",
    "ang liit mo", "ang liit liit mo", "murag bata ka",
    "murag kindergarten", "murag grade one pa",
    "ang taas mo murag poste", "taas na taas",

    # ── Teeth bullying — added ───────────────────────────────
    "ngipon mo murag mais", "ngipon mo murag pader", "bungi ka",
    "bungal ka", "dilaw ang ngipon", "bulok ang ngipon",
    "wala kay ngipon", "ngipon mo murag saging",

    # ── Hair bullying — added ────────────────────────────────
    "kulot ang buhok mo", "mala ang buhok mo", "buhok mo murag walis",
    "buhok mo murag pugad", "walang ayos ang buhok", "malanding buhok",

    # ── Hygiene bullying — added ─────────────────────────────
    "baho ka", "baho mo", "maarte ang amoy mo", "hindi ka naliligo",
    "dili ka naligo", "amoy pawis ka", "amoy ka", "ang baho mo",
    "baho ang katawan mo", "hugaw ka",

    # ── Mocking laughter phrases — added ─────────────────────
    "hala bira", "kadaog nimo", "kalami nimo", "grabi ka",
    "sus ka", "ay sus", "hala ka", "nganong ingon ana ka",

    # ── Social appearance mocking — added ────────────────────
    "baduy ang damit mo", "baduy ka", "luma ang damit mo",
    "ukay ukay lang", "hindi branded", "wala kang style",
    "jologs ka", "jejemon ka", "jologs ang damit",
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
    "buang", "ulaga",                       # Grade 6 additions
    "napaka bobo", "ang tanga", "wala kang kwenta",  # sentence additions
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
    "bungal", "unano", "taas kaayo", "gamay kaayo",  # Grade 6 additions
    "ugly", "pangit", "big nose", "flat nose",       # sentence additions
    "bucktooth", "four eyes", "pimple face",
    # ── Appearance-shaming additions (Grade 6 face/body/skin/hair) ──
    "tambok", "taba", "mataba", "baboy", "fat", "chubby",
    "ilong", "itom", "mata", "pandak", "putot", "liit",
    "ngipon", "bulok", "dilaw", "buhok", "kulot", "mala", "nawong",
    "murag baboy", "murag unggoy", "baduy",
}

BODY_KEYWORDS = {
    "tambok", "tambokikoy", "baboy", "putot", "pandak",
    "niwang", "butong", "murag palito", "murag sundang",
    "dako og tiyan", "murag dwende", "murag nuno",
    "murag baka", "murag litson", "mataba", "payat",
    "parang stick", "parang buntis",
    "buto buto", "kalansay",                # Grade 6 additions
    "fat", "fatty", "fatso", "skinny",      # sentence additions
    "too short", "too tall", "smelly", "amoy", "baho",
    # ── Body-shaming additions (Grade 6 weight/size) ──
    "taba", "chubby", "taas", "liit", "buto",
}

EMOTIONAL_KEYWORDS = {
    "hilak nasad", "iiyak na yan", "pikon", "ampon",
    "wala kang kwenta", "walay gustong", "crybaby", "loser",
    "hilakon", "sgeg hilak", "luod kaayo ka", "cringe",
    "bida bida", "oa kaayo", "gidat ugan",
    "walay amigo", "wala kay barkada",      # Grade 6 additions
    "iyakin", "nobody likes you", "wala kang kaibigan",  # sentence additions
    "freak", "weirdo",
}

THREAT_KEYWORDS = {
    "patyon", "kill", "sumbagay", "away ta",
    "suwayi", "papatayin", "mamamatay",
    "bungoan tika", "suntukan ta", "sampalan tika",  # Grade 6 additions
    "kill yourself", "go kill yourself", "patay ka na sana",  # sentence additions
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
    # FIX 1 — removed single-word "dakog"/"dakug" → "dakog ilong" mappings.
    # They injected "ilong" into every dakog phrase ("dakog mata mo" became
    # "dakog ilong mata mo"). Only the multi-word dakog variants below are safe.
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

    # ── Grade 6 Bisaya (Davao) — field-observed, added ─────────────────────
    "buwan":        "buang",
    "bwang":        "buang",
    "buyang":       "buang",
    "bongal":       "bungal",
    "onano":        "unano",

    # ── English contraction / mishearing normalization (added) ─────────────
    # clean_text turns "you're" -> "you re"; map the no-apostrophe spellings to
    # the same spaced form so the "you re ..." terms match. Bare "your" is NOT
    # mapped (too common); only the wrong-grammar 2-word insults are.
    # NOTE: deliberately NO "X" -> "X mo" Tagalog maps — those reintroduce the
    # dakog-style injection bug ("napaka bobo mo" -> "napaka bobo mo mo"), and
    # the base words (bobo/tanga/walang kwenta/pangit) already catch them.
    "youre":        "you re",
    "your stupid":  "you re stupid",
    "your dumb":    "you re dumb",
    "your ugly":    "you re ugly",
    "your fat":     "you re fat",

    # ── Appearance-word mishearings (added) ────────────────────────────────
    # Only genuinely new keys: pure self-maps (tambok→tambok, etc.) and keys
    # already mapped above (tambuk, thambok, tambog, panggo, bunge, bongal) are
    # intentionally omitted (see convention note above).
    "itum":             "itom",
    "eetom":            "itom",
    "doling":           "duling",
    "pooling":          "duling",
    "pundak":           "pandak",
    "pongo":            "pango",
    "bonggi":           "bungi",
    "koulot":           "kulot",
    "baaduy":           "baduy",
    "murag babuy":      "murag baboy",
    "murag ungoy":      "murag unggoy",
    "murag unggo":      "murag unggoy",
    "mukha kang babuy": "mukha kang baboy",
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
    has_soft      = len(soft_hits) >= 1
    is_casual     = len(laughing) > 0
    # FIX 3/4 — a single soft hit now counts as profanity DETECTION so the
    # audio-primary path (FIX 4) can confirm it by tone/YAMNet. The old "2+ soft"
    # gate was a TEXT-ONLY false-alarm guard from before audio was wired in; that
    # guard now lives downstream (Track A needs aggressive audio; Track B still
    # requires 2+ soft / repetition — see aggression.process_text Layer 2), so a
    # lone soft word can never alert on the quiet path.
    has_profanity = has_hard or has_soft

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
