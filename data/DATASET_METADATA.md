# RailWay AI: Dataset Disclosure & Provenance Card

## Academic & Provenance Declaration
> **IMPORTANT DISCLOSURE**: This dataset (`railway_questions_dataset.csv`) contains **curated and synthetically formulated passenger inquiries** modeled after public Indian Railways passenger reservation rules, IRCTC ticketing guidelines, and common traveler scenarios. 
> 
> **It is NOT official or private CRIS/IRCTC passenger log data.** It does not contain private passenger booking records, PII (Personally Identifiable Information), or confidential railway transaction databases. All questions and scenarios were authored for academic training, benchmarking, and machine learning demonstration purposes.

---

## Dataset Summary
- **Total Samples**: 360 inquiries
- **Classes / Categories**: 6 balanced classes (60 samples each)
- **Language**: English (including colloquial Indian Railway terms: *Tatkal, GNWL, RLWL, PQWL, TDR, RAC, PNR, Sleeper, 3AC, Vande Bharat, Rajdhani, PRS, TTE*)
- **Target Variable**: `category` (Categorical string label)
- **Input Feature**: `question` (Natural language inquiry string)

---

## Categories & Coverage Breakdown

| Category | Sample Count | Scope & Real-World Domain Grounding |
| :--- | :--- | :--- |
| **WAITLIST** | 60 | GNWL, RLWL, PQWL, confirmation chances, waitlist auto-cancellation rules, charting impact, offline counter WL vs e-ticket WL regulations. |
| **RAC** | 60 | Reservation Against Cancellation berth allocation (Side Lower sharing), bedroll eligibility, chart movement, TTE berth allocation rules, travel legality. |
| **PNR_STATUS** | 60 | Checking 10-digit PNR status, chart preparation timelines (1st chart 4 hrs prior, 2nd chart 30 mins prior), berth codes (LB, MB, UB, SL, SU), status codes (CNF, RAC, WL). |
| **CANCELLATION** | 60 | Cancellation charge slabs (before 48 hrs, 48–12 hrs, 12–4 hrs), post-charting cancellation, Tatkal non-refundable rules, TDR filing guidelines, train delay/cancellation. |
| **REFUND** | 60 | Refund settlement timelines (3–7 working days), payment gateway reconciliation, auto-refunds for dropped waitlist tickets, clerkage deductions (Rs. 60), TDR claims. |
| **BOOKING_ISSUES** | 60 | Bank account debited but ticket not booked, Tatkal opening hours (10 AM AC / 11 AM Non-AC), IRCTC session timeouts, Captcha errors, Aadhaar verification booking limits. |

---

## Data Split Protocol (Leakage-Free)
- **Train Split**: 70% (252 samples)
- **Validation Split**: 15% (54 samples)
- **Test Split**: 15% (54 samples)
- **Partitioning Strategy**: Stratified Split by Category (`stratify=y`) with fixed seed `42`.
- **Vocabulary Fit**: The text vocabulary and token frequency dictionaries are constructed **strictly from the Training Set** only. Out-of-vocabulary tokens in Validation and Test sets are mapped to the special `<UNK>` token to ensure zero data leakage.
