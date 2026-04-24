# scripts/create_contracts.py
"""Generate 25 realistic legal contracts for the RAG pipeline."""

import os

os.makedirs("data/raw", exist_ok=True)

CONTRACT_TEMPLATES = {
    "SOFTWARE_LICENSE": """SOFTWARE LICENSE AGREEMENT

This Software License Agreement is entered into as of {date},
by and between {company_a} ("Licensor") and {company_b} ("Licensee").

1. GRANT OF LICENSE
Subject to the terms of this Agreement, Licensor grants Licensee a non-exclusive,
non-transferable license to use the Software solely for internal business purposes.

2. RESTRICTIONS
Licensee shall not: (a) copy, modify, or distribute the Software; (b) reverse engineer
or decompile the Software; (c) sublicense or sell the Software; (d) use the Software
for any unlawful purpose or in violation of applicable laws.

3. INTELLECTUAL PROPERTY
The Software and all copies are proprietary to Licensor. All rights not specifically
granted are reserved. Licensee acknowledges that no title to intellectual property
in the Software is transferred to Licensee.

4. CONFIDENTIALITY
Each party agrees to maintain in confidence all Confidential Information of the other
party. Neither party shall disclose Confidential Information to third parties without
prior written consent. This obligation survives termination for five years.

5. TERM AND TERMINATION
This Agreement commences on the Effective Date and continues for one year unless
earlier terminated. Either party may terminate upon thirty days written notice.
Licensor may terminate immediately upon material breach by Licensee.

6. WARRANTY DISCLAIMER
THE SOFTWARE IS PROVIDED AS IS WITHOUT WARRANTY OF ANY KIND. LICENSOR DISCLAIMS
ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE.

7. LIMITATION OF LIABILITY
IN NO EVENT SHALL LICENSOR BE LIABLE FOR INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES. LICENSOR TOTAL LIABILITY SHALL NOT EXCEED FEES PAID IN THE PRIOR TWELVE MONTHS.

8. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.

9. ENTIRE AGREEMENT
This Agreement constitutes the entire agreement between the parties and supersedes
all prior agreements, negotiations, and understandings, whether oral or written.
""",
    "CONSULTING_SERVICES": """CONSULTING SERVICES AGREEMENT

This Consulting Services Agreement is made as of {date},
between {company_a} ("Consultant") and {company_b} ("Client").

1. SERVICES
Consultant agrees to provide consulting services described in Statement of Work No. 1.
Consultant shall perform Services professionally and consistent with industry standards.

2. COMPENSATION
Client shall pay Consultant at the rate of two hundred fifty dollars per hour.
Consultant shall submit invoices monthly. Client shall pay all undisputed amounts
within thirty days of receipt of each invoice.

3. INDEPENDENT CONTRACTOR
Consultant is an independent contractor and not an employee of Client. Consultant
is solely responsible for all taxes and insurance. Nothing creates a partnership,
joint venture, agency, or employment relationship between the parties.

4. INTELLECTUAL PROPERTY OWNERSHIP
All work product created by Consultant in performance of Services shall be work
made for hire. To the extent any Work Product does not qualify as work made for
hire, Consultant assigns all right, title, and interest to Client upon full payment.

5. NON-SOLICITATION
During the term and for twelve months thereafter, Consultant shall not solicit or
hire any employee of Client without prior written consent. Client shall not solicit
Consultant employees who were involved in performing the Services.

6. INDEMNIFICATION
Each party shall indemnify and hold harmless the other party from any claims,
damages, losses, and expenses arising from that party's breach of this Agreement
or negligent acts or omissions in connection herewith.

7. TERMINATION
Either party may terminate upon fifteen days written notice to the other party.
Client may terminate immediately for cause if Consultant materially breaches and
fails to cure such breach within five business days of written notice.

8. GOVERNING LAW
This Agreement shall be governed by the laws of the State of New York without
regard to its conflict of law provisions.
""",
    "NONDISCLOSURE": """MUTUAL NONDISCLOSURE AGREEMENT

This Mutual Nondisclosure Agreement is entered into as of {date},
between {company_a} and {company_b} (collectively the "Parties").

1. DEFINITION OF CONFIDENTIAL INFORMATION
Confidential Information means any information disclosed by either Party to the
other, directly or indirectly, in writing or orally, that is designated as
confidential or that reasonably should be understood to be confidential given the
nature of the information and circumstances of disclosure.

2. OBLIGATIONS OF RECEIVING PARTY
Each Party agrees to: (a) hold Confidential Information in strict confidence using
at least the same degree of care used for its own confidential information, but no
less than reasonable care; (b) not disclose to third parties without prior written
consent; (c) use solely for evaluating a potential business relationship.

3. EXCLUSIONS FROM CONFIDENTIAL INFORMATION
Confidential Information does not include information that: (a) is or becomes
publicly known through no breach of this Agreement; (b) was rightfully in the
receiving Party's possession before disclosure; (c) is independently developed by
the receiving Party without use of Confidential Information; (d) is required to be
disclosed by law or court order, provided prompt notice is given to the disclosing Party.

4. RETURN OF INFORMATION
Upon request by the disclosing Party, the receiving Party shall promptly return or
destroy all Confidential Information, including all copies, notes, and summaries,
and certify in writing that it has done so.

5. TERM AND SURVIVAL
This Agreement remains in effect for three years from execution. The obligations
with respect to Confidential Information disclosed during the term shall survive
for five years from the date of initial disclosure.

6. REMEDIES FOR BREACH
The Parties acknowledge that breach of this Agreement may cause irreparable harm
for which monetary damages would be inadequate. Accordingly, either Party may seek
injunctive or other equitable relief without the requirement of posting bond,
in addition to all other remedies available at law or in equity.
""",
    "EMPLOYMENT": """EMPLOYMENT AGREEMENT

This Employment Agreement is entered into as of {date},
between {company_a} ("Employer") and a qualified candidate ("Employee").

1. POSITION AND DUTIES
Employer agrees to employ Employee as Senior Software Engineer. Employee shall
perform all duties reasonably assigned and shall devote full professional time
and attention to the business and affairs of Employer.

2. COMPENSATION
Employer shall pay Employee a base salary of one hundred twenty thousand dollars
per year, payable bi-weekly. Employee shall be eligible for an annual performance
bonus at the discretion of the Board of Directors based on company and individual performance.

3. BENEFITS
Employee shall be entitled to participate in all benefit plans maintained by
Employer, including health insurance, dental insurance, vision insurance, and
401k plan with employer matching, subject to the terms and eligibility requirements
of such plans as they may be amended from time to time.

4. AT-WILL EMPLOYMENT
Employee's employment is at-will and may be terminated by either party at any time,
with or without cause, upon two weeks written notice. Employer may terminate
immediately for cause, including but not limited to theft, fraud, gross misconduct,
or material breach of this Agreement.

5. CONFIDENTIALITY AND TRADE SECRETS
Employee agrees to maintain in strict confidence all proprietary and confidential
information of Employer during employment and thereafter. Employee shall not disclose
trade secrets, customer information, or confidential business information to any
third party without prior written consent.

6. INTELLECTUAL PROPERTY ASSIGNMENT
All inventions, works of authorship, software, and other intellectual property
created by Employee during employment that relate to Employer's business or use
Employer's resources shall be owned exclusively by Employer. Employee hereby
assigns all rights therein to Employer.
""",
    "LEASE": """COMMERCIAL LEASE AGREEMENT

This Commercial Lease Agreement is entered into as of {date},
between {company_a} ("Landlord") and {company_b} ("Tenant").

1. PREMISES
Landlord hereby leases to Tenant the premises located at 123 Business Park Drive,
comprising approximately five thousand square feet of office and warehouse space,
together with the right to use common areas of the building.

2. LEASE TERM
The lease term shall commence on the Effective Date and expire thirty-six months
thereafter, unless sooner terminated pursuant to the terms hereof or extended by
mutual written agreement of the parties.

3. BASE RENT AND ADJUSTMENTS
Tenant shall pay monthly base rent of eight thousand dollars, due on the first day
of each calendar month. Base rent shall increase by three percent annually on each
anniversary of the commencement date. Late payments shall incur a penalty of five
percent of the monthly rent after a five-day grace period.

4. SECURITY DEPOSIT
Tenant shall deposit sixteen thousand dollars as a security deposit upon execution.
Landlord shall return the deposit within thirty days after lease termination, less
any amounts applied to unpaid rent or damages beyond normal wear and tear.

5. USE OF PREMISES AND COMPLIANCE
Tenant shall use the premises solely for general office and light warehouse purposes.
Tenant shall comply with all applicable laws, regulations, and ordinances, and shall
not use the premises for any illegal purpose or activity that would increase
Landlord's insurance premiums.

6. MAINTENANCE, REPAIRS, AND ALTERATIONS
Tenant shall maintain the premises in good condition. Landlord shall be responsible
for structural repairs and maintenance of common areas. Tenant shall not make
alterations without prior written consent of Landlord, which shall not be
unreasonably withheld for non-structural improvements.

7. INSURANCE REQUIREMENTS
Tenant shall maintain commercial general liability insurance with minimum coverage
of two million dollars per occurrence throughout the lease term, naming Landlord
as additional insured, and shall provide certificates of insurance upon request.
""",
}

COMPANIES_A = [
    "TechCorp Inc.",
    "DataSystems LLC",
    "InnovateTech Corp.",
    "CloudBase Inc.",
    "NetSolutions Ltd.",
]
COMPANIES_B = [
    "Enterprise Ltd.",
    "Global Partners LLC",
    "Strategic Corp.",
    "Solutions Inc.",
    "Ventures Group",
]
DATES = [
    "January 1, 2024",
    "February 15, 2024",
    "March 1, 2024",
    "April 10, 2024",
    "May 20, 2024",
]
JURISDICTIONS = ["Delaware", "New York", "California", "Texas", "Illinois"]
TEMPLATE_NAMES = list(CONTRACT_TEMPLATES.keys())

for idx in range(25):
    template_name = TEMPLATE_NAMES[idx % len(TEMPLATE_NAMES)]
    template_text = CONTRACT_TEMPLATES[template_name]
    company_a = COMPANIES_A[idx % len(COMPANIES_A)]
    company_b = COMPANIES_B[idx % len(COMPANIES_B)]
    date = DATES[idx % len(DATES)]
    jurisdiction = JURISDICTIONS[idx % len(JURISDICTIONS)]

    content = template_text.format(
        date=date,
        company_a=company_a,
        company_b=company_b,
    )

    renewal = (
        "Automatic annual renewal unless 60 days written notice given by either party"
        if idx % 2 == 0
        else "Manual renewal required in writing signed by both parties"
    )
    dispute = (
        "Binding arbitration under AAA Commercial Rules"
        if idx % 2 == 0
        else "Exclusive jurisdiction of courts in the governing state"
    )

    schedule = f"""
SCHEDULE A - CONTRACT {idx + 1:02d} SPECIFIC TERMS

Contract Reference Number: CTR-2024-{idx + 1:04d}
Contracting Parties: {company_a} and {company_b}
Effective Date: {date}
Governing Jurisdiction: {jurisdiction}
Renewal Terms: {renewal}
Dispute Resolution: {dispute}
Force Majeure: Neither party shall be liable for delays caused by circumstances
beyond their reasonable control, including acts of God, war, terrorism, pandemic,
government action, or failure of third-party telecommunications infrastructure.
Amendment Procedure: This Agreement may only be amended by written instrument
signed by authorised representatives of both parties. No oral modifications shall
be binding on either party.
Severability: If any provision of this Agreement is found invalid or unenforceable
by a court of competent jurisdiction, the remainder of the Agreement shall continue
in full force and effect as if such provision had never been included.
Waiver: Failure by either party to enforce any provision of this Agreement shall
not constitute a waiver of that party's right to enforce such provision in the future.
Notices: All notices shall be in writing and delivered by certified mail, overnight
courier, or email with delivery confirmation to the addresses set forth herein.
Counterparts: This Agreement may be executed in one or more counterparts, each of
which shall be deemed an original and all of which together shall constitute one
and the same instrument.
"""

    full_content = content + schedule
    filename = f"data/raw/contract_{idx + 1:02d}_{template_name}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"Created {filename} ({len(full_content):,} chars)")

print("\nDone - 25 contracts created in data/raw/")
