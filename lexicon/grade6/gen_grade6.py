"""Grade 6 Bisaya bullying phrase generator for EchoSense.
Edit the word lists / patterns and re-run: python3 gen_grade6.py"""
import csv, random, re
random.seed(6)

# ---------- 1. ADJECTIVE INSULTS: (word, english, category, severity, -a form or None)
ADJ = [
 ("bogo","stupid","intelligence","med","bogoa"),
 ("bugo","stupid","intelligence","med","bugoa"),
 ("buang","crazy","intelligence","med","buanga"),
 ("boang","crazy","intelligence","med","boanga"),
 ("tanga","dumb","intelligence","med","tangaa"),
 ("bobo","dumb","intelligence","med",None),
 ("hungog","foolish","intelligence","med","hungoga"),
 ("abno","weird/abnormal","intelligence","med","abnoa"),
 ("pangit","ugly","appearance","med","pangita"),
 ("ngil-ad","ugly","appearance","med",None),
 ("tambok","fat","appearance","med","tamboka"),
 ("niwang","skinny","appearance","med","niwanga"),
 ("itom","dark-skinned","appearance","high","itoma"),
 ("baho","smelly","appearance","med",None),
 ("hugaw","dirty","appearance","med","hugawa"),
 ("bulingit","filthy","appearance","med","bulingita"),
 ("bungi","gap-toothed","appearance","med",None),
 ("pandak","short","appearance","med","pandaka"),
 ("kutohon","lice-ridden","appearance","med",None),
 ("hilakon","a crybaby","personality","low",None),
 ("pikon","a sore loser","personality","low",None),
 ("sipsip","a suck-up","personality","low",None),
 ("pabibo","a show-off","personality","low",None),
 ("papansin","an attention-seeker","personality","low",None),
 ("hambog","boastful","personality","low","hamboga"),
 ("arte","so pretentious","personality","low",None),
 ("samok","annoying","personality","low","samoka"),
 ("kuripot","stingy","personality","low",None),
 ("sumbungero","a tattletale","personality","med",None),
 ("sumbungera","a tattletale (girl)","personality","med",None),
 ("pobre","poor","poverty","med","pobrea"),
 ("bayot","gay (slur use)","identity","high","bayota"),
 ("lakin-on","mannish (slur use)","identity","high",None),
 ("babayeon","girly (slur use, boys)","identity","high",None),
]
ADJ_PATTERNS = [
 ("{w} ka!","You're {g}!"),
 ("{w} ka uy!","You're {g}, ugh!"),
 ("{w} ka oy!","You're {g}, ugh!"),
 ("{w} kaayo ka.","You're so {g}."),
 ("{w} kaau ka.","You're so {g}."),
 ("{w} gyud ka.","You're really {g}."),
 ("{w} jud ka uy.","You're really {g}."),
 ("Ka-{w} nimo!","You're so {g}!"),
 ("Ka{w} nimo!","You're so {g}!"),
 ("Ka-{w} nimo uy!","You're so {g}!"),
 ("Pagka-{w} gyud nimo.","How {g} you really are."),
 ("Pagka{w} gyud nimo.","How {g} you really are."),
 ("{w} man ka.","You're {g} anyway."),
 ("{w} diay ka.","So you're {g}."),
 ("{w} na pud ka.","You're being {g} again."),
 ("Ngano {w} man kaayo ka?","Why are you so {g}?"),
 ("Uy, {w}!","Hey, {g}!"),
 ("Hoy {w}!","Hey, {g}!"),
 ("{w} siya, oh!","He/she's {g}, look!"),
 ("{w} kaayo siya, hahaha.","He/she's so {g}, haha."),
 ("Kining {w} nga bata.","This {g} kid."),
 ("Ikaw ang pinaka-{w} sa klase.","You're the most {g} in class."),
 ("{w} ka, mao wala kay higala.","You're {g}, that's why you have no friends."),
 ("{w} ka, ayaw ko duola.","You're {g}, don't come near me."),
]
ADJ_A_PATTERNS = [
 ("{a} nimo uy!","You're so {g}!"),
 ("{a} nimo oy!","You're so {g}!"),
 ("Kining {a}!","This {g} one!"),
]

# ---------- 2. NOUN NAME-CALLING: (word, english, category, severity)
NOUN = [
 ("yawa","devil","profanity","high"),("piste","pest/damn","profanity","high"),
 ("animal","animal","profanity","high"),("hayop","beast","profanity","high"),
 ("gago","idiot","profanity","high"),("ulol","idiot","profanity","high"),
 ("bwisit","nuisance","profanity","med"),("baboy","pig","appearance","med"),
 ("unggoy","monkey","appearance","med"),("kalansay","skeleton","appearance","med"),
 ("unano","dwarf","appearance","med"),("negro","negro (colorism)","appearance","high"),
 ("bugoy","delinquent","personality","med"),("buwaya","ball hog/greedy","personality","low"),
 ("duwende","goblin","appearance","med"),("iyakin","crybaby","personality","low"),
]
NOUN_PATTERNS = [
 ("{w} ka!","You {g}!"),("{w} ka uy!","You {g}, ugh!"),("{w}!","{g}!"),
 ("Uy, {w}!","Hey, {g}!"),("Murag {w} ka.","You're like a {g}."),
 ("{w} nga bata!","{g} kid!"),("{w} man diay ka.","So you're a {g}."),
 ("Hoy {w}, hilom diha!","Hey {g}, shut up!"),("{w} kaayo ka.","You're such a {g}."),
 ("Pisteng {w}!","Damn {g}!"),
]

# ---------- 3. THREATS: (active future, passive, english verb, past participle)
VERB = [
 ("bunalan","mabunalan","beat","beaten"),("sumbagon","masumbag","punch","punched"),
 ("sagpaon","masagpa","slap","slapped"),("tamparon","matampal","slap","slapped"),
 ("sipaon","masipa","kick","kicked"),("itulod","matulod","push","pushed"),
 ("patyon","mapatay","kill","killed"),("kuniton","makunit","pinch","pinched"),
 ("labayon","malabay","throw something at","hit"),("bugbogon","mabugbog","beat up","beaten up"),
 ("kulataon","makulata","beat","beaten"),("sabunutan","masabunutan","pull the hair of","hair-pulled"),
]
VERB_PATTERNS = [
 ("{v} tika!","I'll {g} you!"),("{v} ta ka!","I'll {g} you!"),
 ("{v} tika unya.","I'll {g} you later."),("{v} tika sa gawas.","I'll {g} you outside."),
 ("{v} tika sa uwian.","I'll {g} you at dismissal."),("{v} gyud tika!","I'll really {g} you!"),
 ("Ugma, {v} tika.","Tomorrow I'll {g} you."),("{v} tika, tan-awa ra.","I'll {g} you, just watch."),
 ("Gusto ka {p}?","You want to get {pp}?"),("Basin {p} ka.","You might get {pp}."),
 ("Kung mosumbong ka, {v} tika.","If you snitch, I'll {g} you."),
 ("Ayaw ko pilia kay {v} tika.","Don't mess with me or I'll {g} you."),
 ("Hilom diha kay {v} tika.","Shut up or I'll {g} you."),
 ("{v} tika kung dili nimo ihatag imong baon.","I'll {g} you if you don't hand over your allowance."),
]

# ---------- 4. GRADE 6 FIXED PHRASES (exclusion, extortion, teasing, snitching, etc.)
FIXED = [
 ("exclusion","med","Bati na ta!","We're not friends anymore!"),
 ("exclusion","med","Dili na ko makig-friend nimo.","I won't be your friend anymore."),
 ("exclusion","med","Ayaw siya apila sa dula.","Don't let him/her join the game."),
 ("exclusion","med","Out ka sa among grupo.","You're out of our group."),
 ("exclusion","med","Dili ka namo friend.","You're not our friend."),
 ("exclusion","med","Walay gusto mo-group nimo.","Nobody wants to group with you."),
 ("exclusion","med","Ayaw siya tabi-i.","Don't sit next to him/her."),
 ("exclusion","med","Ayaw siya istoryahi.","Don't talk to him/her."),
 ("exclusion","med","Balhin didto, ayaw diri.","Move over there, not here."),
 ("exclusion","med","Kinsa man nagdapit nimo?","Who invited you?"),
 ("exclusion","med","Layo diha, kadiri ka.","Go away, you're gross."),
 ("exclusion","med","Ayaw ko hikapa, naa kay kuto.","Don't touch me, you have lice."),
 ("extortion","high","Hatagi ko sa imong baon.","Give me your allowance."),
 ("extortion","high","Pila imong baon? Akoa na lang.","How much is your allowance? I'll take it."),
 ("extortion","med","Palita ko og snacks sa canteen.","Buy me snacks at the canteen."),
 ("extortion","high","Ilibre ko kay kung dili, bati ta.","Treat me or we're not friends."),
 ("extortion","med","Pakopyaha ko sa imong assignment.","Let me copy your assignment."),
 ("extortion","med","Ikaw mo-buhat sa akong project.","You'll do my project."),
 ("extortion","high","Akoa na ning imong ballpen.","Your pen is mine now."),
 ("extortion","med","Ikaw mag-cleaners para nako.","You do my cleaning duty."),
 ("extortion","high","Ugma magdala ka og kwarta.","Tomorrow you bring money."),
 ("snitching","high","Ayaw sumbong sa maestra ha.","Don't tell the teacher, okay."),
 ("snitching","high","Ayaw sumbong ni ma'am.","Don't tell ma'am."),
 ("snitching","high","Isumbong ka? Sige, tan-awa ra.","You'll tell? Go ahead, just watch."),
 ("snitching","med","Sumbungero! Mosumbong sa mama.","Tattletale! Running to mama."),
 ("snitching","high","Kung mosumbong ka, bati ta ug patay ka.","If you tell, we're not friends and you're dead."),
 ("mockery","low","Ayeee, crush nimo siya no?","Ooh, you have a crush on him/her, right?"),
 ("mockery","low","Uyab nimo siya! Ayeee!","He/she's your boyfriend/girlfriend! Ooh!"),
 ("mockery","med","Mama's boy man ka.","You're a mama's boy."),
 ("mockery","med","Mohilak na siya, oh!","Look, he/she's going to cry!"),
 ("mockery","med","Nag-ihi siya sa iyang karsones!","He/she peed his/her pants!"),
 ("mockery","med","Bisaya kaayo ka mo-English.","Your English is so Bisaya."),
 ("mockery","med","Wala kay baon, pobre.","You have no allowance, poor kid."),
 ("mockery","med","Gisi imong sapatos, hahaha.","Your shoes are torn, haha."),
 ("mockery","med","Pareho ra imong uniporme matag adlaw.","Same uniform every day."),
 ("mockery","med","Bagsak na pud ka sa quiz, bogo.","You failed the quiz again, dummy."),
 ("mockery","low","Feeling nimo gwapo ka?","You think you're handsome?"),
 ("mockery","low","Feeling nimo gwapa ka?","You think you're pretty?"),
 ("mockery","med","Wala kay papa, no?","You don't have a dad, right?"),
 ("mockery","med","Imong papa hubog man.","Your dad's a drunk."),
 ("provocation","high","Unsa man, ha? Away ta?","What, huh? Want to fight?"),
 ("provocation","high","Unsa'y problema nimo?","What's your problem?"),
 ("provocation","high","Imong mama!","Your mom! (fight starter)"),
 ("provocation","high","Yawa imong mama!","Damn your mom!"),
 ("provocation","high","Abangan tika sa gawas.","I'll wait for you outside."),
 ("provocation","high","Hulat lang ka sa uwian.","Just wait till dismissal."),
 ("provocation","high","Ipabunal tika sa akong kuya.","I'll have my big brother beat you."),
 ("property","high","Ilabay nako imong bag.","I'll throw your bag."),
 ("property","high","Gisi-on nako imong notebook.","I'll tear your notebook."),
 ("property","high","Tagoan nako imong tsinelas.","I'll hide your slippers."),
 ("online","med","Ayaw siya i-add sa GC.","Don't add him/her to the group chat."),
 ("online","med","I-kick siya sa GC.","Kick him/her from the GC."),
 ("online","high","I-post nako imong pic para kataw-an ka.","I'll post your pic so people laugh at you."),
]

# ---------- 5. COMBOS: opener + insult + follow-up (how it's said in a real burst)
OPENERS = ["Uy,","Hoy,","Unsa man?","Tan-awa ni,","Hahaha,","Yawa,","Piste,"]
FOLLOWUPS = ["hahaha.","ayaw ko duola.","wala kay higala.","hilom diha.","layo diha.",
             "ayaw sumbong ha.","abangan tika sa gawas.","bunalan tika unya.",
             "out ka sa among grupo.","mohilak na siya, oh."]

rows=[]
ENGLISH_FIXES=[
 (r'\bso so ','so '),(r'\bThis an? ','This '),(r'\bHow an? ','How '),
 (r'\bbeat up you\b','beat you up'),(r'\bpull the hair of you\b','pull your hair'),
 (r'\ba ([aeiouAEIOU])',r'an \1'),
]
def fix_english(eng):
    for pat,new in ENGLISH_FIXES: eng=re.sub(pat,new,eng)
    return eng
def add(cat,sev,pattern,sent,eng,base):
    eng=re.sub(r'\b(so|most|really|being|Hey,) an? ',r'\1 ',eng)
    eng=fix_english(eng)
    rows.append([cat,sev,pattern,sent[0].upper()+sent[1:],eng,base])

for w,g,c,s,a in ADJ:
    for p,e in ADJ_PATTERNS: add(c,s,p,p.format(w=w),e.format(g=g),w)
    if a:
        for p,e in ADJ_A_PATTERNS: add(c,s,p.replace("{a}","{a}"),p.format(a=a),e.format(g=g),w)
for w,g,c,s in NOUN:
    for p,e in NOUN_PATTERNS:
        if p.startswith("Pisteng") and w=="piste": continue
        add(c,s,p,p.format(w=w),e.format(g=g),w)
for v,pv,g,pp in VERB:
    for p,e in VERB_PATTERNS: add("threat","high",p,p.format(v=v,p=pv),e.format(g=g,pp=pp),v)
for c,s,t,e in FIXED: add(c,s,"fixed",t,e,"")
cores=[(w,g,c,s) for w,g,c,s,_ in ADJ]
for _ in range(300):
    o=random.choice(OPENERS); w,g,c,s=random.choice(cores); f=random.choice(FOLLOWUPS)
    add("combo",s if "tika" not in f else "high","opener+core+followup",
        f"{o} {w} kaayo ka, {f}",f"[{o}] you're so {g}, [{f}]",w)

seen=set(); out=[]
for r in rows:
    if r[3] in seen: continue
    seen.add(r[3]); out.append(r)
with open("grade6_bisaya_bullying.csv","w",newline="",encoding="utf-8") as f:
    wr=csv.writer(f); wr.writerow(["id","category","severity","pattern","bisaya","english","base_word"])
    for i,r in enumerate(out,1): wr.writerow([i]+r)
from collections import Counter
print(len(out)); print(Counter(r[0] for r in out))
