"""One coarse label space shared by every dataset and by the engine.

Each benchmark uses its own label vocabulary (ai4privacy says GIVENNAME,
Gretel says first_name, Blacken says PERSON). Comparing them requires mapping
all of them onto the same small set of classes. The classes are chosen around
what a redaction tool must catch, not around how any one recognizer is
organised - e.g. every government-issued identifier is GOV_ID regardless of
country, because a reviewer cares that it was blacked out, not which country
recognizer claimed it.

Two special classes:
  * IGNORE - real annotations that are outside Blacken's declared scope
    (usernames, passwords, API keys, PINs, geo-coordinates). They are neither
    rewarded nor penalised: a prediction overlapping one is not a false
    positive, and missing one is not a false negative. Reported separately
    so the scope decision stays visible rather than silently flattering the
    numbers.
  * ORGANIZATION - company names are not personal data under GDPR/HIPAA, and
    Blacken deliberately shows them unticked. They are excluded from the
    headline "core PII" metrics and reported on their own.
"""

from typing import Optional

CORE_CLASSES = (
    "PERSON",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "DATE",
    "GOV_ID",
    "FINANCIAL",
    "OTHER_ID",
    "NETWORK",
    "AGE",
)
ORG = "ORGANIZATION"
IGNORE = "IGNORE"

# --- Blacken engine output -> class ---------------------------------------
ENGINE_TO_CLASS = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "ADDRESS": "ADDRESS",
    "LOCATION": "ADDRESS",
    "ZIP_CODE": "ADDRESS",
    "DATE_TIME": "DATE",
    "AGE": "AGE",
    "US_SSN": "GOV_ID",
    "US_ITIN": "GOV_ID",
    "US_PASSPORT": "GOV_ID",
    "US_DRIVER_LICENSE": "GOV_ID",
    "UK_NHS": "GOV_ID",
    "UK_NINO": "GOV_ID",
    "IN_AADHAAR": "GOV_ID",
    "IN_PAN": "GOV_ID",
    "IN_PASSPORT": "GOV_ID",
    "IN_VOTER": "GOV_ID",
    "IN_VEHICLE_REGISTRATION": "GOV_ID",
    "IN_GSTIN": "GOV_ID",
    "GOV_ID": "GOV_ID",
    "PASSPORT_NUMBER": "GOV_ID",
    "DRIVER_LICENSE": "GOV_ID",
    "TAX_ID": "GOV_ID",
    "NATIONAL_ID": "GOV_ID",
    "UK_POSTCODE": "ADDRESS",
    "ABA_ROUTING_NUMBER": "FINANCIAL",
    "ROUTING_NUMBER": "FINANCIAL",
    "UPI_ID": "FINANCIAL",
    "CUSTOMER_ID": "OTHER_ID",
    "POLICY_NUMBER": "OTHER_ID",
    "ID_NUMBER": "OTHER_ID",
    "CREDIT_CARD": "FINANCIAL",
    "IBAN_CODE": "FINANCIAL",
    "US_BANK_NUMBER": "FINANCIAL",
    "ACCOUNT_NUMBER": "FINANCIAL",
    "CRYPTO": "FINANCIAL",
    "SWIFT_CODE": "FINANCIAL",
    "MEDICAL_RECORD_NUMBER": "OTHER_ID",
    "PATIENT_ID": "OTHER_ID",
    "MEDICAL_LICENSE": "OTHER_ID",
    "EMPLOYEE_ID": "OTHER_ID",
    "IP_ADDRESS": "NETWORK",
    "MAC_ADDRESS": "NETWORK",
    "URL": "NETWORK",
    "ORGANIZATION": ORG,
    # NRP (nationality/religion/political group) and UUID are not identifying
    # on their own; mapping them to IGNORE means they never count as hits or
    # false positives.
    "NRP": IGNORE,
    "UUID": IGNORE,
}

# --- ai4privacy pii-masking-400k -> class ---------------------------------
AI4PRIVACY_TO_CLASS = {
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
    "EMAIL": "EMAIL",
    "TELEPHONENUM": "PHONE",
    "CITY": "ADDRESS",
    "STREET": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "ZIPCODE": "ADDRESS",
    "DATEOFBIRTH": "DATE",
    "SOCIALNUM": "GOV_ID",
    "IDCARDNUM": "GOV_ID",
    "TAXNUM": "GOV_ID",
    "DRIVERLICENSENUM": "GOV_ID",
    "CREDITCARDNUMBER": "FINANCIAL",
    "ACCOUNTNUM": "FINANCIAL",
    "USERNAME": IGNORE,
    "PASSWORD": IGNORE,
}

# --- Gretel synthetic_pii_finance_multilingual -> class -------------------
GRETEL_TO_CLASS = {
    "name": "PERSON",
    "first_name": "PERSON",
    "last_name": "PERSON",
    "email": "EMAIL",
    "phone_number": "PHONE",
    "street_address": "ADDRESS",
    "date": "DATE",
    "date_of_birth": "DATE",
    "date_time": "DATE",
    "time": "DATE",
    "ssn": "GOV_ID",
    "passport_number": "GOV_ID",
    "driver_license_number": "GOV_ID",
    "credit_card_number": "FINANCIAL",
    "bban": "FINANCIAL",
    "iban": "FINANCIAL",
    "bank_routing_number": "FINANCIAL",
    "swift_bic_code": "FINANCIAL",
    "credit_card_security_code": "FINANCIAL",
    "employee_id": "OTHER_ID",
    "customer_id": "OTHER_ID",
    "ipv4": "NETWORK",
    "ipv6": "NETWORK",
    "company": ORG,
    "account_pin": IGNORE,
    "api_key": IGNORE,
    "password": IGNORE,
    "user_name": IGNORE,
    "local_latlng": IGNORE,
}


def engine_class(entity_type: str) -> Optional[str]:
    """Unknown engine types are treated as core PII of their own name rather
    than dropped, so a newly added recognizer still shows up in the report."""
    return ENGINE_TO_CLASS.get(entity_type, entity_type)
