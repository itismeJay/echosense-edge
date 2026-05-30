ENGLISH_BLACKLIST = [
    "stupid", "idiot", "shut up", "hate you",
    "kill you", "fight me", "loser", "dumb",
    "ugly", "freak", "moron", "jerk",
    "tang ina", "tangina", "worthless", "nobody likes you",
    "go away", "you smell", "get lost", "weirdo",
    "crybaby", "nobody wants you", "you're nothing", "get out"
]

FILIPINO_BLACKLIST = [
    "putangina", "putang ina", "gago", "gaga",
    "bobo", "tanga", "hayop", "pakyu",
    "ulol", "inutil", "puta", "leche",
    "hinayupak", "lintik", "buwisit",
    "pangit", "pandak", "mataba", "payat",
    "bakla", "abnormal", "baliw", "wala kang kwenta",
    "walang kwenta", "walang silbi", "tang ina mo",
    "ampota", "pakshet", "putcha", "p*ta"
]

BISAYA_BLACKLIST = [
    "buang", "yawa", "bogo", "uwang",
    "boang", "bilat", "punyeta", "hudas",
    "iyot", "pisti", "atay", "buyag",
    "animal", "ngano", "bastos",
    "pangit", "tonto", "ungo", "piste",
    "buanga", "maot", "walay pulos", "katawa", "luoy"
]

ALL_BLACKLIST = ENGLISH_BLACKLIST + FILIPINO_BLACKLIST + BISAYA_BLACKLIST

def contains_blacklisted_word(text: str) -> bool:
    text_lower = text.lower()
    for word in ALL_BLACKLIST:
        if word in text_lower:
            return True
    return False

def get_detected_words(text: str) -> list:
    text_lower = text.lower()
    detected = []
    for word in ALL_BLACKLIST:
        if word in text_lower:
            detected.append(word)
    return detected