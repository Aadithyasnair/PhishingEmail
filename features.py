import re
import numpy as np
import unicodedata


class PhishingFeatureExtractor:
    def __init__(self, blocklist=None):
        # Expanded homoglyph candidates can be added here; we also detect non-ascii characters
        self.urgency_words = [
            "urgent", "immediately", "suspend", "24 hours", "account limited",
            "verify now", "action required", "security alert", "unauthorized",
            "act now", "don't delay", "final notice",
            # additional variants useful for payroll/HR notices
            "required", "required update", "mandatory", "deadline", "log in", "login", "sign in", "action needed"
        ]
        self.sensitive_brands = ["paypal", "microsoft", "google", "apple", "amazon", "bank", "rbi", "sbi"]
        # include HR/payroll-related sensitive tokens that often appear in targeted phishing
        self.sensitive_brands += ["payroll", "hr", "human resources", "employee", "benefits"]
        self.blocklist = blocklist if blocklist else set()

    def _count_non_ascii_and_confusables(self, text):
        non_ascii = 0
        confusable = 0
        for ch in text:
            if ord(ch) > 127:
                non_ascii += 1
                # simple heuristic: if character's name mentions CYRILLIC or GREEK, count as confusable
                try:
                    name = unicodedata.name(ch).lower()
                    if 'cyrillic' in name or 'greek' in name:
                        confusable += 1
                except Exception:
                    pass
        return non_ascii, confusable

    def extract_features(self, text, sender=None, subject=None, return_flags=False):
        # text: full email body; sender: raw From header; subject: Subject header
        if not isinstance(text, str):
            text = str(text)
        text_lower = text.lower()
        sender_lower = (sender or '').lower()
        subject_lower = (subject or '').lower()
        features = []
        critical_flags = []

        # 1. Non-ascii and homoglyph-like characters
        non_ascii_count, confusable_count = self._count_non_ascii_and_confusables(text)
        features.append(non_ascii_count)
        if confusable_count > 0:
            critical_flags.append("Homoglyph Detected (Cyrillic/Greek chars resembling Latin)")

        # 2. Mixed alphanumeric tokens (letter-digit-letter to avoid matching times like '3pm')
        mixed_tokens = re.findall(r"\b[A-Za-z]+\d+[A-Za-z]+\b", text)
        mixed_count = len(mixed_tokens)
        features.append(mixed_count)
        if mixed_count > 0:
            critical_flags.append("Mixed Alpha-Numeric Detected")

        # 3. URL Analysis
        urls = re.findall(r"https?://[\w\-\.%:/?=&]+", text)
        features.append(len(urls))

        shorteners = ["bit.ly", "goo.gl", "tinyurl", "ow.ly", "t.co", "is.gd", "buff.ly"]
        has_shortener = 0
        for url in urls:
            if any(s in url for s in shorteners):
                has_shortener = 1
        features.append(has_shortener)

        if has_shortener:
            critical_flags.append("URL Shortener Detected")

        has_ip_in_url = 0
        for url in urls:
            if re.search(r"\d{1,3}(?:\.\d{1,3}){3}", url):
                has_ip_in_url = 1
        features.append(has_ip_in_url)

        if has_ip_in_url:
            critical_flags.append("IP Address In URL")

        # Blocklist check
        in_blocklist = 0
        if self.blocklist:
            for url in urls:
                for bad_url in self.blocklist:
                    if bad_url in url or url in bad_url:
                        in_blocklist = 1
                        critical_flags.append(f"Blacklisted URL Found: {bad_url}")
                        break
        features.append(in_blocklist)

        # 4. Urgency and brand mentions
        urgency_count = sum(1 for w in self.urgency_words if w in text_lower)
        features.append(urgency_count)
        if urgency_count > 0:
            critical_flags.append(f"Urgency Words Found: {urgency_count}")

        brand_count = sum(1 for b in self.sensitive_brands if b in text_lower)
        features.append(brand_count)
        if brand_count > 0:
            critical_flags.append(f"Brand Mentions: {brand_count}")

        # 5. Caps ratio and punctuation
        caps_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)
        features.append(caps_ratio)

        has_html = 1 if re.search(r"<[^>]+>", text) else 0
        features.append(has_html)

        # 6. Finance keywords
        finance_words = ["upi", "refund", "kyc", "pan", "income tax", "otp", "irs", "invoice"]
        fin_count = sum(1 for w in finance_words if w in text_lower)
        features.append(fin_count)
        if fin_count > 0:
            critical_flags.append("Finance-related Keywords Detected")

        # 7. Domain-level non-ascii in URLs
        domain_nonascii = 0
        # additional URL/domain heuristics
        login_like = 0
        suspicious_domain_pattern = 0
        for url in urls:
            # extract domain
            m = re.match(r"https?://([^/]+)", url)
            if m:
                domain = m.group(1)
                if any(ord(ch) > 127 for ch in domain):
                    domain_nonascii = 1
                    critical_flags.append("Non-ASCII Characters In Domain")
                    break
                # detect login/auth-like paths or domains
                low = url.lower()
                if any(token in low for token in ["/auth", "/login", "login", "/signin", "sign-in", "/authenticate"]) or any(tok in domain.lower() for tok in ["auth", "login", "signin"]):
                    login_like = 1
                # suspicious domain patterns: multiple hyphens or 'benefits'/'portal' tokens
                if domain.count('-') >= 2 or any(tok in domain.lower() for tok in ["benefits","portal","hr","payroll","employee"]):
                    suspicious_domain_pattern = 1
                    critical_flags.append(f"Suspicious domain pattern: {domain}")
        features.append(domain_nonascii)
        features.append(login_like)
        features.append(suspicious_domain_pattern)

        # 8. Sender checks
        # parse sender domain if present
        sender_domain_nonascii = 0
        sender_brand_in_domain = 0
        if sender:
            # try to extract email address
            m = re.search(r"([\w\.-]+@[^\s>]+)", sender)
            if m:
                addr = m.group(1)
                parts = addr.split('@')
                if len(parts) == 2:
                    dom = parts[1]
                    if any(ord(ch) > 127 for ch in dom):
                        sender_domain_nonascii = 1
                        critical_flags.append("Non-ASCII Characters In Sender Domain")
                    # check for known brands in domain (impersonation)
                    for b in self.sensitive_brands:
                        if b in dom:
                            sender_brand_in_domain += 1
                            critical_flags.append(f"Sender Domain Mentions Brand: {b}")
                            break
        features.append(sender_domain_nonascii)
        features.append(sender_brand_in_domain)

        # 9. Subject-level features
        subject_urgency = sum(1 for w in self.urgency_words if w in subject_lower)
        features.append(subject_urgency)
        if subject_urgency > 0:
            critical_flags.append(f"Subject Urgency Words: {subject_urgency}")

        subject_fin = sum(1 for w in ["upi","refund","invoice","otp","kyc","pan"] if w in subject_lower)
        features.append(subject_fin)
        if subject_fin > 0:
            critical_flags.append("Subject Contains Financial Keywords")

        subject_nonascii = 1 if any(ord(ch) > 127 for ch in subject or '') else 0
        features.append(subject_nonascii)

        # Additional composite heuristic: payroll/login link with urgency
        if urls:
            payroll_terms = ["payroll","employee","salary","benefits","hr","human resources"]
            if any(t in text_lower for t in payroll_terms) and (urgency_count > 0 or login_like or suspicious_domain_pattern):
                critical_flags.append("Payroll/HR login link with urgency — suspicious")
                # note: we add only to critical_flags; feature vector length remains unchanged
        if return_flags:
            return np.array(features), critical_flags
        return np.array(features)

    def get_feature_names(self):
        return [
            "NonAsciiCharCount", "MixedAlphaNumeric", "UrlCount", "HasShortener",
            "HasIpUrl", "UrlInBlocklist", "UrgencyWordCount", "BrandMentionCount",
            "CapsRatio", "HasHtml", "FinanceKeywords", "DomainNonAscii",
            "LoginLikeUrl", "SuspiciousDomainPattern", "SenderDomainNonAscii", "SenderBrandInDomain",
            "SubjectUrgencyCount", "SubjectFinanceCount", "SubjectNonAscii"
        ]
