"""Closed vocabularies used to down-weight spaCy LOCATION guesses that are
not personally identifying.

HIPAA Safe Harbor treats geography at the state level and above as safe to
disclose; only smaller subdivisions (street, city, ZIP) identify someone.
spaCy tags countries and states LOCATION at the same flat confidence as
cities and streets, so without this a document's "United Kingdom" or "TX"
is pre-ticked for redaction exactly like a home address. These are
down-weighted (still shown for review, not pre-selected), never dropped.
"""

COUNTRIES = {
    "afghanistan", "albania", "algeria", "argentina", "armenia", "australia", "austria",
    "azerbaijan", "bahrain", "bangladesh", "belarus", "belgium", "bhutan", "bolivia",
    "bosnia", "botswana", "brazil", "bulgaria", "cambodia", "cameroon", "canada", "chile",
    "china", "colombia", "costa rica", "croatia", "cuba", "cyprus", "czech republic",
    "czechia", "denmark", "ecuador", "egypt", "estonia", "ethiopia", "fiji", "finland",
    "france", "georgia", "germany", "ghana", "greece", "guatemala", "hong kong", "hungary",
    "iceland", "india", "indonesia", "iran", "iraq", "ireland", "israel", "italy", "jamaica",
    "japan", "jordan", "kazakhstan", "kenya", "kuwait", "laos", "latvia", "lebanon", "libya",
    "lithuania", "luxembourg", "malaysia", "maldives", "malta", "mauritius", "mexico",
    "moldova", "mongolia", "morocco", "mozambique", "myanmar", "namibia", "nepal",
    "netherlands", "the netherlands", "new zealand", "nigeria", "north korea", "norway",
    "oman", "pakistan", "panama", "paraguay", "peru", "philippines", "poland", "portugal",
    "qatar", "romania", "russia", "rwanda", "saudi arabia", "senegal", "serbia",
    "singapore", "slovakia", "slovenia", "somalia", "south africa", "south korea", "korea",
    "spain", "sri lanka", "sudan", "sweden", "switzerland", "syria", "taiwan", "tanzania",
    "thailand", "tunisia", "turkey", "uganda", "ukraine", "united arab emirates", "uae",
    "united kingdom", "great britain", "britain", "england", "scotland", "wales",
    "northern ireland", "united states", "united states of america", "america", "usa",
    "uruguay", "uzbekistan", "venezuela", "vietnam", "yemen", "zambia", "zimbabwe",
    "the united states", "the united kingdom", "u.s.", "u.k.", "u.s.a.", "europe", "asia",
    "africa", "north america", "south america", "latin america", "middle east",
    "european union", "eu", "emea", "apac",
}

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york state", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington state", "west virginia", "wisconsin", "wyoming",
}

INDIAN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal", "jammu and kashmir", "ladakh",
}

OTHER_REGIONS = {
    "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest",
    "midwest", "the midwest", "west coast", "east coast", "ontario", "quebec",
    "british columbia", "alberta", "new south wales", "victoria", "queensland",
}

NON_IDENTIFYING_PLACES = COUNTRIES | US_STATES | INDIAN_STATES | OTHER_REGIONS


def is_non_identifying_place(text: str) -> bool:
    t = text.strip().strip(".,").lower()
    if t in NON_IDENTIFYING_PLACES:
        return True
    # Bare 2-3 letter ALL-CAPS codes: state/country abbreviations (TX, UK,
    # GB, ENG, USA) and acronyms spaCy mislabels (MD, RN, VIN).
    stripped = text.strip().replace(".", "")
    return stripped.isalpha() and stripped.isupper() and len(stripped) <= 3
