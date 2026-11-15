"""Name recognition for Indian-format documents, plus guards that stop common
non-name words (tech stacks, institutions, form labels) being tagged as people
or places.

Why this exists (measured, see evals/data/indian_gold.txt): the small spaCy
pipeline and GLiNER-edge recognise only ~53% of person-name tokens in Indian
KYC forms, affidavits, payslips, tickets and OCR'd IDs, and close to 0% when
the document is in ALL CAPS. They also label short Indian first names
("Disha", "Madhu") as LOCATION or ORGANIZATION, and tag resume tech words
("React", "PyMuPDF") as places.

Everything here is a closed, document-independent vocabulary or a structural
rule - nothing is fitted to a test document:

* `IndianNameRecognizer` - runs of capitalised words made of known Indian
  first names / surnames / initials, a "First Surname" pair whose surname is
  known or has a distinctively Indian ending (-iya, -kar, -wala, -jee ...),
  unambiguous single first names, and the name line at the top of a document.
* `CONTEXT_NAME_RECOGNIZERS` - names after an honorific (Mr, Smt, Shri, Late),
  after a relation marker (S/o, W/o, D/o, Son of, Wife of ...), and after a
  field label with no colon, as OCR produces ("Name RAJU SHARMA").
* helpers used by the engine to correct entity types and drop tech terms.
"""

import re
from typing import List

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult

# --------------------------------------------------------------------------
# Vocabularies (lower-case)
# --------------------------------------------------------------------------
_FIRST = """
aarav aarti aarush aashish aayush abdul abhay abhijit abhilash abhinav abhishek abid aditi aditya
afreen afsana afzal ahmad ahmed aishwarya ajay ajit ajmal akash akhil akhilesh akshara akshay
alka alok amaan aman amandeep amar amarjeet amarnath ambika amina amir amit amita amitabh amol
amrita anand ananya anil anirban anirudh anita anjali anju ankit ankita ankur anmol annapurna
anshu anshul anuj anupam anupama anurag anuradha anushka anwar aparna archana arif arjun arpita
arshad arti arun aruna arvind arya asha ashish ashok ashwin asif asim aslam atul avinash ayaan
ayesha azhar babita babu badal balaji balasubramanian balbir baldev balwinder bharat bhavana bhavesh
bhavna bhavya bhupinder bhushan bijoy bikram bindu brijesh chaitanya chanda chandan chandra chandrashekhar
charan chetan chhaya chitra daljeet daksh darshan deepa deepak deepali deepika deepti dev devendra
devika dhananjay dharmendra dhruv dilip dimple dinesh disha divya durga ekta faisal farhan farida
farooq fatima feroz gagan gagandeep ganesh gaurav gauri gayatri geeta geetha girish gita gopal
gopi govind gulshan gurdeep gurmeet gurpreet gyan hamid hardik hari harinder haris harish harjinder
harjit harpreet harsh harsha harshad harshit harshita hemant hema hemlata himanshu hina hitesh
imran indira indu inderjeet irfan ishaan ishan ishita jagdish jagjit jai jaideep jasleen jaspreet
jaswinder jatin javed jaya jayant jayesh jeevan jitendra jyoti kabir kailash kajal kalpana kamal
kamala kamini kanika kapil karan karthik kartik kashish kavita kavya keshav ketan khushboo kirti
kishore komal krishna krishnan kriti kuldeep kunal kusum lakshmi lakshya lalit lalita lalitha lata
laxmi leela madhavi madhu madhur madhuri mahendra mahesh malini mallika mallikarjun mamta manav
mandeep manish manisha manju manjunath manoj mansi meena meenakshi megha mehul mohammed mohammad
mohan mohd mohit monika mukesh mukul murali muskan nadia nafisa nagendra naina namita nandini
narendra naresh nasreen naveen navneet nazia neelam neeraj neetu neha nidhi nikhil nikita nilesh
nirmal nirmala nisha nitin nitish padma padmavathi pallavi pankaj paramjit parul parveen pawan
payal pooja poonam prabhat pradeep prakash pramod pranav prashant pratibha pratik praveen prem
preeti prerna priya priyanka puja punit purushottam pushpa rachna radha radhika raghav
raghu rahim rahul raj rajan rajat rajeev rajendra rajesh rajinder rajiv rajni raju rakesh rakhi
ram ramesh rameshwar rana ranbir randeep ranjana ranjit rashmi ratan ravi ravinder rekha renu
reshma revathi riddhi rinku rishabh rishi ritesh ritika ritu riya rohan rohit roshni ruchi rupa
rupinder sachin sadhana sagar sahil sai sajid sakshi salim salma sameer samir sana sandeep sandhya
sangeeta sanjay sanjeev sanjana santosh sapna sarita saroj satish satyam saurabh savita seema
shabana shabnam shafiq shahid shailesh shalini shankar sharad sharmila shashank shashi sheetal
shilpa shiv shivam shobha shraddha shreya shrikant shruti shubham shweta siddharth siddhi simran
sita smita sneha sohail sonal sonia sonu sophia subhash sudha sudhanshu sudhir sujata sukhwinder
sumit sunaina sundar sunil sunita suraj surekha suresh surinder sushant sushil sushma swati
tanvi tanuja tarun tejas thiruvengadam trupti tushar uday ujjwal uma umesh urmila usha uttam
vaibhav vandana varsha varun vasant vedant venkat venkatesh vidya vijay vijaya vikas vikram
vimal vinay vineet vinod vipin vipul viraj vishal vishnu vishwas vivek yamini yash yashwant
yogesh yusuf zahid zainab zara zeeshan zoya
"""

_SURNAMES = """
acharya adhikari advani agarwal aggarwal agnihotri ahluwalia ahmed ahuja ali ansari arora
awasthi bajpai bajwa bakshi banerjee bansal basu batra bedi bhagat bhalla bhandari bhardwaj
bhaskar bhatia bhatnagar bhatt bhattacharya bhattacharjee bhosale bisht bose brar chakraborty
chandra chatterjee chaturvedi chaudhary chaudhari chauhan chawla cheema chopra dalal das dasgupta
datta deol desai deshmukh deshpande devi dhawan dhillon dixit dubey dutta dwivedi gandhi ganguly
garg gill goel goswami gowda grewal gupta hegde hussain iyengar iyer jain jaiswal jha joshi kakkar
kapoor kaur khan khanna khurana kohli kothari krishnamurthy kulkarni kumar kundu lal mahajan
malhotra malik mane mathur mehra mehta menon merchant mishra mittal mukherjee mukhopadhyay
murthy naidu nair nambiar narayanan nath negi pandey pandit parekh patel pathak pathan patil
pillai prasad qureshi raghunathan rajput ramachandran ramaswamy rana rao rathore rawat reddy
saini saxena sen sengupta seth shah shaikh sharma sheikh shetty shinde shukla siddiqui sinha
singh sodhi soni srinivasan subramaniam subramanian sundaram suri syed tandon thakur
thakkar tiwari tripathi trivedi tyagi upadhyay vaidya varma verma vyas yadav sandhu
gokhale apte bapat chitnis karnik nadkarni phadke ranade sathe sawant gaikwad jadhav more pawar
salunkhe kamble waghmare lokhande bhattacharyya chowdhury mondal mandal ghosh pal roy saha sarkar
mukhopadhyaya gangopadhyay chattopadhyay bandyopadhyay majumdar mazumdar biswas dey haldar
krishnan venkataraman venkatesan raman natarajan rajagopal sankaran ganesan murugan palaniappan
chettiar pillay namboodiri kurup panicker varghese kurian thomas mathew george
chowdary raju goud kamath pai prabhu bhat nayak
kini shenoy kotadiya panchal parmar solanki chavda
rathod makwana vaghela zala jadeja gohil chudasama barot modi dave pandya
oza raval mistry suthar prajapati darji lad mewada thaker kotecha ruparel majithia
chheda gala sanghvi doshi parikh kapadia mody vora bohra lakhani chandwani lalwani
mirchandani hiranandani sippy kalra sethi anand khattar mann sidhu
virk randhawa sekhon kang hundal aulakh bains dhaliwal
qadri rizvi naqvi zaidi jafri abbasi farooqui usmani khatri baig mirza begum khatun bano
"""

FIRST_NAMES = frozenset(_FIRST.split())
SURNAMES = frozenset(_SURNAMES.split())
NAME_WORDS = FIRST_NAMES | SURNAMES

# First names that are also ordinary words, deities, festivals or places, so a
# lone occurrence is not evidence of a person (still fine inside a full name).
AMBIGUOUS_SINGLE = frozenset(
    """
    asha babu badal bharat chanda chandra deepa dev durga ganesh gayatri gita hari hema indira
    jai jaya kamal kamala krishna lakshmi lata laxmi madhur mamta nirmal prem puja radha raj
    ram rana sagar sai shankar shiv sita sonu sophia sundar uma usha vasant vishnu yash ali
    """.split()
) | SURNAMES

# Distinctively Indian surname endings (Ladhaniya, Tendulkar, Bharuchwala,
# Advani, Mukherjee, Gurappa, Ramaswamy, Krishnamurthy ...).
_SURNAME_SUFFIX = re.compile(
    r"(?:iya|iyah|wala|walla|kar|ani|jee|rjee|appa|swamy|swami|murthy|moorthy|aiah|ayya|bhai|ben|preet|jeet|inder|deep)$"
)

_ABBREVIATIONS = {"mohd", "md", "jr", "sr"}

HONORIFICS = r"Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Shri|Shree|Sri|Smt|Sh|Kumari|Km|Late|Master|Adv|Er|Capt|Col|Maj|Lt"
_HONORIFIC_WORDS = frozenset(h.lower() for h in HONORIFICS.split("|"))

# Words that end a name or mean the capitalised run is an institution, a
# place or a form label rather than a person.
INSTITUTION_WORDS = frozenset(
    """
    road rd marg chowk nagar colony enclave layout sector circle bazaar bazar market mall plaza
    temple mandir masjid gurudwara gurdwara dargah church cathedral university college school
    institute vidyalaya academy hospital hospitals clinic nursing bank airport port station
    stadium bhavan bhawan sadan society trust foundation association samiti sangh limited ltd pvt
    private llp inc corp corporation company co industries enterprises traders trading textiles
    brothers bros sons foods motors auto laboratories labs pharma pharmaceuticals pharmacy
    infotech technologies technology tech solutions systems software services consultancy
    consulting consultants jewellers jewelers sweets stores store exports imports agency agencies
    travels tours hotel hotels restaurant cafe bakery airlines airways railway railways petroleum
    steel cements cement power energy finance financial capital ventures group global international
    national express river lake hills chaturthi jayanti puja navami diwali holi eid
    """.split()
)

LABEL_WORDS = frozenset(
    """
    name names father father's mother mother's husband husband's wife wife's son daughter spouse
    guardian nominee applicant candidate holder member passenger patient employee customer client
    student address residence resident residing phone mobile contact email dob date birth sex gender
    male female age aged occupation designation department branch account signature sign photo
    valid licence license policy number no id card government india republic issued issue
    deponent witness tenant landlord remitter payee bill to from dear hi hello thanks regards
    sincerely cheers present absent leave total amount net pay summary report certificate form
    """.split()
)

# Tech / product vocabulary that spaCy's small model tags as PERSON, LOCATION
# or ORGANIZATION on resumes and engineering documents. Closed list: none of
# these is a personal name or a place someone lives (words that are also
# given names or surnames - Ruby, Swift, Claude, Jenkins - are deliberately
# left out).
TECH_TERMS = frozenset(
    """
    python java javascript typescript golang kotlin scala php perl c++ c# react reactjs react.js
    next.js nextjs node.js nodejs vue vue.js angular svelte vite webpack django flask fastapi nestjs
    tailwind tailwindcss shadcn bootstrap redux graphql docker kubernetes k8s terraform ansible vercel
    netlify railway heroku aws gcp azure firebase supabase cloudflare nginx linux ubuntu debian windows
    macos android ios postgres postgresql mysql sqlite mongodb redis kafka rabbitmq elasticsearch
    snowflake bigquery pgvector chromadb pinecone weaviate qdrant faiss airflow dbt spark hadoop
    databricks tableau looker powerbi excel grafana prometheus datadog sentry kibana postman selenium
    cypress jest pytest pandas numpy scipy matplotlib sklearn scikit-learn pytorch tensorflow keras
    huggingface transformers langchain langgraph langfuse llamaindex openai gemini gpt chatgpt llama
    mistral ollama spacy presidio pymupdf tesseract ocr opencv streamlit gradio jupyter colab git
    github gitlab bitbucket jira confluence notion slack figma canva trello linkedin portfolio nextauth
    resend telegram whatsapp discord twilio stripe razorpay paytm phonepe api apis sdk llm llms rag
    mcp adk nlp ml ai ui ux html css sql nosql json yaml devops mlops ci cd saas b2b chatbot backend
    frontend fullstack google microsoft bot
    """.split()
)

_RELATION = (
    r"S/o|D/o|W/o|C/o|H/o|S\.O\.|D\.O\.|W\.O\.|Son\s+of|Daughter\s+of|Wife\s+of|Husband\s+of|"
    r"Father\s+of|Mother\s+of|Care\s+of|Spouse\s+of"
)

# One name word: Title-case or ALL-CAPS, or an initial ("K."). Never a label,
# honorific, relation word or institution word. A trailing full stop belongs
# to the word only for initials and a few abbreviations ("Mohd."), so a name
# never runs across a sentence end.
_STOP_ALT = "|".join(
    sorted(
        {w for w in LABEL_WORDS | INSTITUTION_WORDS | _HONORIFIC_WORDS if re.fullmatch(r"[a-z]+", w)},
        key=len,
        reverse=True,
    )
)
_NW = (
    rf"(?!(?i:(?:{_STOP_ALT}|of|and|the)\b))"
    rf"(?:[A-Z]\.|(?:Mohd|Md|Jr|Sr)\.|[A-Z][a-zA-Z'\-]+(?!\.\s*[A-Z]{{2}}))"
)
_NAME_RUN = rf"{_NW}(?: {_NW}){{0,3}}"
# A run followed by an institution word is an institution ("Sri Venkateswara
# Temple", "Dr. Reddy's Laboratories"), not a person.
_NOT_INSTITUTION_NEXT = rf"(?! (?i:(?:{'|'.join(sorted(INSTITUTION_WORDS, key=len, reverse=True))})\b))"

_FLAGS = re.MULTILINE

HONORIFIC_NAME = PatternRecognizer(
    supported_entity="PERSON",
    name="HonorificNameRecognizer",
    patterns=[
        Pattern(
            name="honorific_name",
            regex=(
                rf"(?<=(?<![A-Za-z])(?i:{HONORIFICS})\.? +(?:(?i:{HONORIFICS})\.? +)*)"
                rf"{_NAME_RUN}\b{_NOT_INSTITUTION_NEXT}"
            ),
            score=0.8,
        )
    ],
    global_regex_flags=_FLAGS,
)

RELATION_NAME = PatternRecognizer(
    supported_entity="PERSON",
    name="RelationNameRecognizer",
    patterns=[
        Pattern(
            name="relation_name",
            regex=(
                rf"(?<=(?<![A-Za-z])(?i:{_RELATION})\.? *(?:(?i:{HONORIFICS})\.? +)*)"
                rf"{_NAME_RUN}\b{_NOT_INSTITUTION_NEXT}"
            ),
            score=0.85,
        )
    ],
    global_regex_flags=_FLAGS,
)

# Field labels followed by a name with or without a colon - OCR flattens ID
# cards and forms ("Name RAJU SHARMA Son of ...", "Applicant Name Fatima Begum").
_FIELD_LABELS = (
    r"Name|Full\s+Name|Applicant\s+Name|"
    r"Name\s+of\s+(?:the\s+)?(?:Account\s+Holder|Applicant|Student|Candidate|Employee|Patient|Customer|Holder|Member|Insured)|"
    r"Account\s+Holder|Card\s+Holder|Licen[cs]e\s+Holder|Holder|Member|Candidate|Passenger(?:\s+\d+)?|"
    r"Husband'?s\s+Name|Wife'?s\s+Name|Father'?s\s+Name|Mother'?s\s+Name|Guardian'?s\s+Name|"
    r"Bill\s+To|Ship\s+To|Invigilator|Referred\s+by|Policy\s+holder|Insured|Nominee|Payee|Remitter|"
    r"Witness(?:\s+\d+)?|Reporting\s+Manager|Manager|Supervisor|Approved\s+by|Verified\s+by|"
    r"Prepared\s+by|Signed\s+by|Contact\s+Person|Authori[sz]ed\s+Signatory"
)
FIELD_NAME = PatternRecognizer(
    supported_entity="PERSON",
    name="FieldNameRecognizer",
    patterns=[
        Pattern(
            name="field_name",
            regex=(
                rf"(?<=(?<![A-Za-z])(?i:{_FIELD_LABELS})(?: *[:\-] *| +)(?:(?i:{HONORIFICS})\.? +)*)"
                rf"{_NAME_RUN}\b{_NOT_INSTITUTION_NEXT}"
            ),
            score=0.75,
        )
    ],
    global_regex_flags=_FLAGS,
)

CONTEXT_NAME_RECOGNIZERS = [HONORIFIC_NAME, RELATION_NAME, FIELD_NAME]


# --------------------------------------------------------------------------
# Token helpers (also used by the engine)
# --------------------------------------------------------------------------
def _norm(token: str) -> str:
    t = token.strip(".,;:()[]\"").lower()
    return t[:-2] if t.endswith("'s") else t


def is_initial(token: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]\.?", token))


def is_known_name(token: str) -> bool:
    return _norm(token) in NAME_WORDS


def has_indian_surname_shape(token: str) -> bool:
    t = _norm(token)
    return len(t) >= 6 and t.isalpha() and bool(_SURNAME_SUFFIX.search(t)) and t not in TECH_TERMS


def is_tech_or_code(token: str) -> bool:
    """`React`, `Next.js`, `PyMuPDF`, `C++`, `pgvector` - software words, not
    people or places. CamelCase counts, except name prefixes (McDonald,
    DeSouza, MacArthur, O'Brien)."""
    t = token.strip(",;:()[]\"")
    if t.lower() in TECH_TERMS:
        return True
    if re.search(r"[0-9+#]|\.[a-z]", t):
        return True
    if re.search(r"[a-z][A-Z]", t) and not re.match(r"(?:Mc|Mac|De|Di|Da|La|Le|Van|Von|O')[A-Z]", t):
        return True
    return False


def is_institution(token: str) -> bool:
    return _norm(token) in INSTITUTION_WORDS


def all_tokens_known_names(text: str) -> bool:
    tokens = text.split()
    return (
        bool(tokens)
        and all(is_known_name(t) or is_initial(t) for t in tokens)
        and any(is_known_name(t) for t in tokens)
    )


# --------------------------------------------------------------------------
# Gazetteer / structure recognizer
# --------------------------------------------------------------------------
_RUN = re.compile(r"(?:[A-Z][A-Za-z'\-]*\.?)(?: (?:[A-Z][A-Za-z'\-]*\.?))*")


def _is_cased_word(token: str) -> bool:
    core = token.rstrip(".")
    return bool(core) and core[0].isupper() and (
        core.isupper() or core[1:].islower() or "'" in core or "-" in core
    )


class IndianNameRecognizer(EntityRecognizer):
    """Structural name finder backed by Indian first-name / surname lists."""

    def __init__(self, supported_language: str = "en"):
        super().__init__(
            supported_entities=["PERSON"],
            name="IndianNameRecognizer",
            supported_language=supported_language,
        )

    def load(self) -> None:  # nothing to load
        pass

    def analyze(self, text: str, entities: List[str], nlp_artifacts=None) -> List[RecognizerResult]:
        results: List[RecognizerResult] = []
        first_line = self._first_content_line(text)
        for m in _RUN.finditer(text):
            tokens = [
                (t.group(), m.start() + t.start(), m.start() + t.end())
                for t in re.finditer(r"\S+", m.group())
            ]
            for seg in self._segments(self._split_sentences(tokens)):
                found = self._score_segment(seg, text, first_line)
                if found:
                    start, end, score = found
                    results.append(self._result(start, end, score))
        return results

    @staticmethod
    def _first_content_line(text: str):
        offset = 0
        for line in text.split("\n"):
            if line.strip():
                lead = len(line) - len(line.lstrip())
                return (offset + lead, offset + len(line.rstrip()))
            offset += len(line) + 1
        return None

    @staticmethod
    def _split_sentences(tokens):
        """"Kavya. Sudhanshu" is two sentences: drop the full stop from a
        non-initial word and insert a break (None) after it."""
        out = []
        for word, s, e in tokens:
            core = word.rstrip(".")
            if word.endswith(".") and len(core) > 1 and core.lower() not in _ABBREVIATIONS:
                out.append((core, s, s + len(core)))
                out.append(None)
            else:
                out.append((word, s, e))
        return out

    @staticmethod
    def _segments(tokens):
        """Split a capitalised run at sentence ends, honorifics, labels,
        institutions, tech words and mixed-case junk; keep the pieces."""
        seg = []
        for tok in tokens:
            if tok is None:
                if seg:
                    yield seg, None
                seg = []
                continue
            word = tok[0]
            n = _norm(word)
            breaker = (
                n in _HONORIFIC_WORDS
                or n in LABEL_WORDS
                or is_institution(word)
                or is_tech_or_code(word)
                or not (_is_cased_word(word) or is_initial(word))
            )
            if breaker:
                if seg:
                    yield seg, tok
                seg = []
            else:
                seg.append(tok)
        if seg:
            yield seg, None

    def _score_segment(self, seg_and_next, text, first_line):
        seg, breaker_after = seg_and_next
        if breaker_after is not None and is_institution(breaker_after[0]):
            return None  # "Shiv Nadar University", "Indira Gandhi International"
        start, end = seg[0][1], seg[-1][2]
        if self._institution_before(text, start):
            return None
        words = [t[0] for t in seg]
        known = [is_known_name(w) for w in words]
        initials = [is_initial(w) for w in words]
        shaped = [has_indian_surname_shape(w) for w in words]
        strong = [k or s for k, s in zip(known, shaped)]
        unknown = [not (st or i) for st, i in zip(strong, initials)]

        if len(words) >= 2 and not any(unknown) and sum(strong) >= 1:
            if sum(strong) >= 2 or (sum(strong) == 1 and any(initials)):
                return start, end, 0.75 if len(words) >= 3 else 0.7
        # "Jay Ladhaniya", "Vishwas Deshpande": an unrecognised first word
        # followed by a known surname or an unmistakably Indian surname.
        if (
            2 <= len(words) <= 3
            and sum(unknown) == 1
            and unknown[0]
            and len(words[0].rstrip(".")) >= 2
            and (_norm(words[-1]) in SURNAMES or shaped[-1])
        ):
            return start, end, 0.65
        # The name line that opens a resume / letter.
        if (
            first_line is not None
            and start >= first_line[0]
            and end <= first_line[1]
            and text[first_line[0] : first_line[1]].strip() == text[start:end]
            and 2 <= len(words) <= 4
            and any(strong)
        ):
            return start, end, 0.7
        # Lone, unambiguous first name ("Sudhanshu followed you", "Thanks, Disha").
        if len(words) == 1:
            n = _norm(words[0])
            if n in FIRST_NAMES and n not in AMBIGUOUS_SINGLE and len(n) >= 4:
                return start, end, 0.62
        return None

    @staticmethod
    def _institution_before(text: str, start: int) -> bool:
        """"Bank of Baroda"-style: an institution word right before the run
        on the same line ("Hotel Rajdhani", "Hospital Ramesh ...")."""
        window = text[max(0, start - 20) : start]
        if "\n" in window:
            window = window.rsplit("\n", 1)[1]
        prev = window.rstrip().split(" ")[-1] if window.strip() else ""
        return _norm(prev) in {"bank", "hotel", "hospital", "university", "college", "temple"}

    def _result(self, start: int, end: int, score: float) -> RecognizerResult:
        return RecognizerResult(
            entity_type="PERSON",
            start=start,
            end=end,
            score=score,
            analysis_explanation=None,
            recognition_metadata={
                RecognizerResult.RECOGNIZER_NAME_KEY: self.name,
                RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
            },
        )


# --------------------------------------------------------------------------
# Span clean-up used by the engine
# --------------------------------------------------------------------------
_LEADING_HONORIFIC = re.compile(rf"^(?:(?:{HONORIFICS})\.?\s+)+", re.IGNORECASE)
_RELATION_SPLIT = re.compile(rf"\s+(?:{_RELATION})\.?\s+", re.IGNORECASE)
_TRAILING_RELATION_LETTER = re.compile(r"\s+[SDWCH]$")
# Field separators, and any token containing a digit.
_FIELD_BREAK = re.compile(r"\s*[/|;()\[\]]\s*|\s*\S*\d\S*\s*")


def clean_person_span(text: str, start: int, end: int) -> List[tuple]:
    """Split "RAJU SHARMA Son of Mohan Sharma" into two names, cut a model
    span at field separators and numbers ("Ashwath Hegdekatte / Male / 29"),
    drop leading/trailing form labels ("Applicant ..."), a leading honorific
    ("Shri K. Narayanan" -> "K. Narayanan") and a dangling relation letter
    ("Lakshmi Narayanan W" before "/o"). Returns [(start, end)]."""
    pieces = []
    span = text[start:end]
    cursor = 0
    for m in _RELATION_SPLIT.finditer(span):
        pieces.append((start + cursor, start + m.start()))
        cursor = m.end()
    pieces.append((start + cursor, end))

    split = []
    for s, e in pieces:
        cur = s
        for m in _FIELD_BREAK.finditer(text, s, e):
            split.append((cur, m.start()))
            cur = m.end()
        split.append((cur, e))

    out = []
    for s, e in split:
        while s < e and text[s] in " \t,.":
            s += 1
        while e > s and text[e - 1] in " \t,":
            e -= 1
        if s >= e:
            continue
        words = text[s:e].split()
        if all(_norm(w) in LABEL_WORDS for w in words):
            continue
        while len(words) > 1 and _norm(words[0]) in LABEL_WORDS:
            s = s + text[s:e].index(words[0]) + len(words[0])
            while s < e and text[s] == " ":
                s += 1
            words = text[s:e].split()
        piece = text[s:e]
        lead = _LEADING_HONORIFIC.match(piece)
        if lead and lead.end() < len(piece):
            s += lead.end()
            piece = text[s:e]
        if text[e : e + 2] in ("/o", "/O"):
            trail = _TRAILING_RELATION_LETTER.search(piece)
            if trail:
                e = s + trail.start()
        # Strip trailing label words swallowed by a model ("Sharma Son").
        words = text[s:e].split()
        while len(words) > 1 and _norm(words[-1]) in LABEL_WORDS:
            e = s + text[s:e].rstrip().rfind(" ")
            words = text[s:e].split()
        if e > s and text[s:e].strip():
            out.append((s, e))
    return out
