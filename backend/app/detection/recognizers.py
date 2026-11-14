"""Custom Presidio recognizers for identifiers not covered by Presidio's
built-ins, kept from the original app's domain coverage (medical records,
patient/account IDs, ages) but as small, explicit, testable patterns instead
of a giant ad hoc regex dict."""

import re

from presidio_analyzer import Pattern, PatternRecognizer

# These use a lookbehind for the label so only the identifier itself
# (not "MRN-" / "Patient ID:" / ...) is captured as the entity span - the
# labels aren't sensitive, so we shouldn't redact them too. Presidio uses
# the `regex` package under the hood, which (unlike stdlib `re`) supports
# variable-length lookbehind.
MEDICAL_RECORD_NUMBER = PatternRecognizer(
    supported_entity="MEDICAL_RECORD_NUMBER",
    name="MedicalRecordNumberRecognizer",
    patterns=[
        Pattern(
            name="mrn",
            regex=r"(?<=\bMRN[-#:\s]{0,3})\d{6,}\b",
            score=0.85,
        ),
    ],
    context=["medical", "record", "mrn", "chart"],
)

PATIENT_ID = PatternRecognizer(
    supported_entity="PATIENT_ID",
    name="PatientIdRecognizer",
    patterns=[
        Pattern(
            name="patient_id",
            regex=r"(?<=\b(?:Patient\s*ID|PAT\s*#|PT\s*#)[-:\s]{0,3})\d{4,}\b",
            score=0.85,
        ),
    ],
    context=["patient", "id"],
)

ACCOUNT_NUMBER = PatternRecognizer(
    supported_entity="ACCOUNT_NUMBER",
    name="AccountNumberRecognizer",
    patterns=[
        Pattern(
            name="account_number",
            regex=r"(?<=\b(?:Account|Acct)\s*(?:No\.?|Number|#)?[-:\s]{0,3})\d{6,}\b",
            score=0.8,
        ),
    ],
    context=["account", "acct", "bank"],
)

AGE = PatternRecognizer(
    supported_entity="AGE",
    name="AgeRecognizer",
    patterns=[
        Pattern(
            # "Age: 34", "Age/Sex: 52/M", "aged about 47 years"
            name="age_labeled",
            regex=r"(?<=\b(?:Age(?:\s*/\s*(?:Sex|Gender))?|aged(?:\s+about)?)\s*[:\-]?\s*)\d{1,3}\b",
            score=0.75,
        ),
        Pattern(
            name="age_y_o",
            regex=r"\b\d{1,3}\s?(?:y/o|yo|yrs?\.?\s+old)\b",
            score=0.75,
        ),
        Pattern(
            name="age_years_old",
            regex=r"\b\d{1,3}[-\s](?:years?|yrs?)[-\s]old\b",
            score=0.75,
        ),
    ],
    context=["age", "years", "old"],
)

# Presidio has no general-purpose street address recognizer. spaCy's NER
# catches some addresses as LOCATION/FAC but not reliably, so we keep a
# structured pattern for the common "<number> <street name> <suffix>" shape.
ADDRESS = PatternRecognizer(
    supported_entity="ADDRESS",
    name="StreetAddressRecognizer",
    patterns=[
        Pattern(
            name="street_address",
            regex=(
                r"\b\d{1,6}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,4}\s+"
                r"(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|"
                r"Way|Court|Ct|Circle|Cir|Place|Pl|Terrace|Ter|Highway|Hwy|"
                r"Parkway|Pkwy|Square|Sq)\.?\b"
            ),
            score=0.6,
        ),
    ],
    context=["address", "street", "live", "resides", "located"],
)

# Bare 5-digit numbers are too ambiguous to flag as PII on their own (order
# numbers, dosages, years...), so only match when explicitly labeled.
# "PIN code" / "Pincode" is the standard term in India and a few other
# countries, alongside "zip code" / "postal code" elsewhere.
ZIP_CODE = PatternRecognizer(
    supported_entity="ZIP_CODE",
    name="ZipCodeRecognizer",
    patterns=[
        Pattern(
            name="zip_code",
            regex=r"(?<=\b(?:zip|postal|pin)\s*code[:\s]{0,3})\d{4,6}(?:-\d{4})?\b",
            score=0.8,
        ),
        Pattern(
            name="pincode_single_word",
            regex=r"(?<=\bpincode[:\s]{0,3})\d{4,6}\b",
            score=0.8,
        ),
    ],
    context=["zip", "postal", "pin", "code"],
)

# Presidio's built-in DateRecognizer only pairs a day with a month *name* if
# they're hyphen-joined ("14-MAR-1988" / "MAR-1988"); the at-least-as-common
# space- or comma-separated form ("14 March 1988", "14th Mar, 1988") has no
# regex pattern at all and falls back entirely on spaCy's statistical NER,
# which isn't guaranteed to catch the full span (e.g. dropping the leading
# day number). Cover it deterministically instead, so an unambiguous
# calendar date reads as one regardless of what the NLP model guesses.
_MONTH_NAME_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|"
    r"July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DATE_TEXTUAL = PatternRecognizer(
    supported_entity="DATE_TIME",
    name="TextualDateRecognizer",
    patterns=[
        Pattern(
            # Scored just above spaCy's flat 0.85 DATE score: when spaCy's
            # NER only catches part of the same span (e.g. "May 2026" out of
            # "11 May 2026"), overlap resolution keeps whichever candidate
            # has the higher confidence - this full, correctly-bounded regex
            # match should always win over a truncated statistical guess.
            name="day_month_year",
            regex=rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_NAME_PATTERN}\.?,?\s+\d{{2,4}}\b",
            score=0.86,
        ),
        Pattern(
            name="month_day_year",
            regex=rf"\b{_MONTH_NAME_PATTERN}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{2,4}}\b",
            score=0.86,
        ),
        Pattern(
            name="day_month",
            regex=rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_NAME_PATTERN}\b",
            score=0.55,
        ),
    ],
    context=["date", "born", "dob", "on", "dated"],
)

# --------------------------------------------------------------------------
# Labeled-field recognizers.
#
# Forms, letters, statements and records almost always introduce personal
# data with a label ("Name:", "Employee Code:", "Address:"). The label is a far
# stronger signal than the value's shape - an employee code like LT482019 or a
# UAN like 100928374615 has no checksum or distinctive format, but after
# "Employee Code:" or "UAN:" it is unambiguous. These recognizers key off the
# label and capture only the value.
#
# Presidio compiles patterns with IGNORECASE by default; these pass explicit
# flags so that capitalisation in the *value* (e.g. a name must start with a
# capital letter) is respected, with `(?i:...)` scoping case-insensitivity to
# the label alone.
# --------------------------------------------------------------------------
_FLAGS = re.MULTILINE
_SEP = r"\s*(?:No\.?|Number|Num|#|Code)?\s*[:#\-]\s*"
_ID_VALUE = r"(?=[A-Z0-9/-]*\d)[A-Z0-9][A-Z0-9/-]{3,24}[A-Z0-9]\b"

_LABELED_ID_FAMILIES = {
    # entity type: label alternatives (case-insensitive)
    "PASSPORT_NUMBER": r"passport",
    "DRIVER_LICENSE": r"driver'?s?\s+licen[cs]e|driving\s+licen[cs]e|\bDL",
    "TAX_ID": r"tax\s+id|TIN|SSN|social\s+security|national\s+insurance|NI",
    "NATIONAL_ID": r"aadhaar|aadhar|national\s+id|id\s+card|voter\s+id|EPIC",
    "ROUTING_NUMBER": r"routing|ABA|sort\s+code",
    "ACCOUNT_NUMBER": r"(?:bank\s+)?a/?c|account",
    "EMPLOYEE_ID": r"employee\s+(?:id|code)|emp\s+(?:id|code)|staff\s+id|badge",
    "CUSTOMER_ID": r"customer\s+id|client\s+id|member\s+id|membership|frequent\s+flyer|loyalty",
    "POLICY_NUMBER": r"policy|claim|insurance\s+id",
    "ID_NUMBER": r"UAN|PF|ESIC|roll|enrol(?:l)?ment|registration|student\s+id|license|licence",
}

LABELED_IDENTIFIERS = [
    PatternRecognizer(
        supported_entity=entity,
        name=f"Labeled{entity.title().replace('_', '')}Recognizer",
        patterns=[
            Pattern(
                name=f"labeled_{entity.lower()}",
                regex=rf"(?<=(?i:\b(?:{labels}))(?:{_SEP}|\s+))(?:{_ID_VALUE}|\d(?:[ -]?\d){{5,17}}\b)",
                score=0.7,
            )
        ],
        global_regex_flags=_FLAGS,
    )
    for entity, labels in _LABELED_ID_FAMILIES.items()
]

_NAME_LABELS = (
    r"name|patient|applicant|co-applicant|beneficiary|insured(?:\s+person)?|"
    r"passenger|client|student|employee|tenant|landlord|owner|deponent|"
    r"guardian|parent(?:/guardian)?|emergency\s+contact|witness|complainant|"
    r"attending|physician|doctor|nominee|account\s+(?:holder|name)|"
    r"father'?s\s+name|mother'?s\s+name|spouse(?:'s\s+name)?|remitter|"
    r"payee|signed\s+by|prepared\s+by"
)
_HONORIFIC = r"(?:(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Shri|Smt|Sri|Kumari)\.?\s+)?"
_NAME_WORD = r"[A-Z][a-zA-Z'\-]+\.?"

LABELED_PERSON = PatternRecognizer(
    supported_entity="PERSON",
    name="LabeledPersonRecognizer",
    patterns=[
        Pattern(
            name="labeled_person",
            regex=(
                rf"(?<=(?i:\b(?:{_NAME_LABELS}))\s*:\s*{_HONORIFIC})"
                # Single spaces only: forms separate fields with runs of
                # spaces/tabs ("Aarav Joshi   Roll No: ..."), never names.
                rf"{_NAME_WORD}(?: {_NAME_WORD}){{0,3}}"
            ),
            score=0.8,
        )
    ],
    global_regex_flags=_FLAGS,
)

# Everything after an address label up to the end of that field: a line
# break, a run of 2+ spaces, a pipe, or the next "Label:" on the same line.
# A negative lookbehind keeps "Email Address:" / "IP Address:" out - those
# have dedicated recognizers.
LABELED_ADDRESS = PatternRecognizer(
    supported_entity="ADDRESS",
    name="LabeledAddressRecognizer",
    patterns=[
        Pattern(
            name="labeled_address",
            regex=(
                r"(?<=(?<!(?i:e-?mail|ip|mac|web|ipv[46])\s)"
                r"(?i:\b(?:address|addr|residing\s+at|resides\s+at|residential\s+address|"
                r"correspondence\s+address|home\s+address|delivery\s+address|"
                r"permanent\s+address|billing\s+address|premises\s+at)\b)\s*[:\-]?\s*)"
                r"(?=[^\n]*\d)[A-Za-z0-9#][^\n|]{4,140}?"
                # Field ends at: wide gap, pipe, next "Label:", sentence end,
                # end of line - or where prose resumes ("..., do hereby",
                # "... for 11 months"), so the rest of the sentence survives.
                r"(?=\s{2,}|\s*\||\.?[ \t]+(?:[A-Z][a-z]+[ \t]?){1,3}:|\s*\.\s|\.?\s*$|"
                r",?[ \t]+(?:for|to|with|from|since|and|which|where|who|do|does|hereby|"
                r"is|are|was|were|has|have|will|shall|until)\b)"
            ),
            score=0.7,
        )
    ],
    global_regex_flags=_FLAGS,
)

# Indian addresses rarely end in a Western street suffix; they are built
# from locality keywords ("Shanti Nagar Society", "Sector 21", "3rd Cross")
# introduced by a unit marker ("Flat 3C", "B-204", "Plot 17", "H.No 5").
_IN_UNIT = r"(?:(?i:flat|house|h\.?\s?no\.?|plot|door|shop|bldg|building|unit|apt\.?|no\.)\s*[\w/-]*\d[\w/-]*|[A-Z]-?\d{1,4})"
_IN_LOCALITY = (
    r"(?i:nagar|society|soc\.?|sector|colony|apartments?|enclave|layout|cross|"
    r"main|block|phase|marg|chowk|bazaar|bazar|lane|road|street|park|vihar|"
    r"puram|wadi|pada|gali|mohalla|town|extension|ext\.?|residency|heights|tower)"
)
INDIAN_ADDRESS = PatternRecognizer(
    supported_entity="ADDRESS",
    name="IndianAddressRecognizer",
    patterns=[
        Pattern(
            name="in_address",
            regex=(
                rf"\b{_IN_UNIT},?(?:[ \t]+[\w.'/-]+,?){{0,6}}?[ \t]+[\w.'-]*{_IN_LOCALITY}\b"
                r"(?:[ \t]+\d{1,4}[A-Za-z]?\b)?(?:,?[ \t]+[A-Z][\w.'-]+){0,2}"
            ),
            score=0.65,
        )
    ],
    global_regex_flags=_FLAGS,
)

# A 6-digit Indian PIN code directly after a capitalised place name
# ("Kolkata 700016", "Ahmedabad - 380009"), or a US ZIP directly after a
# two-letter state code ("Austin, TX 78704"). The anchoring place/state is
# what makes a bare 5-6 digit number safe to flag.
POSTAL_CODE_ANCHORED = PatternRecognizer(
    supported_entity="ZIP_CODE",
    name="AnchoredPostalCodeRecognizer",
    patterns=[
        Pattern(
            name="in_pin_after_city",
            regex=r"(?<=\b[A-Z][a-z]{2,}[ \t]*[-–,]?[ \t]*)[1-9]\d{2}[ \t]?\d{3}\b(?![\d,.])",
            score=0.65,
        ),
        Pattern(
            name="us_zip_after_state",
            regex=r"(?<=\b[A-Z]{2},?[ \t]+)\d{5}(?:-\d{4})?\b(?![\d,.])",
            score=0.65,
        ),
    ],
    global_regex_flags=_FLAGS,
)

# UPI virtual payment addresses (name@bank handle) - India's dominant
# payment identifier. Looks like an email without a TLD, so the email
# recognizer never matches it.
UPI_ID = PatternRecognizer(
    supported_entity="UPI_ID",
    name="UpiIdRecognizer",
    patterns=[
        Pattern(
            name="upi",
            regex=(
                r"\b[\w.\-]{2,256}@(?:ok(?:icici|hdfcbank|sbi|axis)|ybl|ibl|axl|paytm|upi|"
                r"apl|ptyes|ptsbi|pthdfc|ptaxis|kotak|icici|hdfcbank|sbi|axisbank|"
                r"barodampay|freecharge|jupiteraxis|fbl|idfcbank)\b(?![.@\w])"
            ),
            score=0.8,
        )
    ],
    context=["upi", "vpa", "pay", "gpay", "phonepe"],
)

# Presidio's PhoneRecognizer only emits numbers libphonenumber validates
# for a known region, which is right for unlabelled digit runs. But a number
# printed right after "Phone:" / "Contact:" / "Helpline:" is a phone number
# to the reader whether or not it validates (toll-free placeholders,
# unusual national formats, OCR-damaged digits) - and leaking it is the
# costly error. Scored below the validated recognizer so a valid match wins.
LABELED_PHONE = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    name="LabeledPhoneRecognizer",
    patterns=[
        Pattern(
            name="labeled_phone",
            regex=(
                r"(?<=(?i:\b(?:phone|ph|tel|telephone|mobile|mob|cell|contact|helpline|"
                r"whatsapp|fax|emergenc(?:y|ies)|call|landline)\b)(?:\s*(?:no\.?|number|#))?"
                r"\s*[:\-]?\s*(?:(?i:on|at|me\s+on)\s+)?)"
                r"\+?\(?\d[\d().\- ]{5,18}\d\b"
            ),
            score=0.65,
        )
    ],
    global_regex_flags=_FLAGS,
)

ALL_CUSTOM_RECOGNIZERS = [
    LABELED_PHONE,
    MEDICAL_RECORD_NUMBER,
    PATIENT_ID,
    ACCOUNT_NUMBER,
    AGE,
    ADDRESS,
    ZIP_CODE,
    DATE_TEXTUAL,
    *LABELED_IDENTIFIERS,
    LABELED_PERSON,
    LABELED_ADDRESS,
    INDIAN_ADDRESS,
    POSTAL_CODE_ANCHORED,
    UPI_ID,
]
