"""The companies in the project, their SEC CIKs and their type.

Types were assigned from each business description only, before any results,
and approved on 16 Nov 2026 (Session 4). They are fixed.
Alphabet is joined with Google Inc. (the same company before the 2015 restructuring).
"""

TYPES = ["Seasonal", "Growth tech", "Cyclical industrial", "FMCG"]

# company -> (type, [(CIK, entity name), ...], one-line reason for the type)
COMPANIES = {
    "Mattel":           ("Seasonal", [("0000063276", "Mattel")],
                         "Toys; most sales ship for the holiday season"),
    "Hasbro":           ("Seasonal", [("0000046080", "Hasbro")],
                         "Toys and games; the same holiday-driven calendar"),
    "Williams-Sonoma":  ("Seasonal", [("0000719955", "Williams-Sonoma")],
                         "Home-furnishings retailer; the holiday quarter is the largest"),
    "Best Buy":         ("Seasonal", [("0000764478", "Best Buy")],
                         "Consumer-electronics retailer; the Nov-Jan quarter dominates"),
    "Macy's":           ("Seasonal", [("0000794367", "Macy's")],
                         "Department stores; the holiday quarter dominates"),
    "Alphabet":         ("Growth tech", [("0001288776", "Google Inc."), ("0001652044", "Alphabet Inc.")],
                         "Online advertising and cloud; revenue has grown most years"),
    "Microsoft":        ("Growth tech", [("0000789019", "Microsoft")],
                         "Software and cloud sold increasingly by subscription"),
    "Salesforce":       ("Growth tech", [("0001108524", "Salesforce")],
                         "Cloud CRM software sold by subscription"),
    "Netflix":          ("Growth tech", [("0001065280", "Netflix")],
                         "Streaming subscriptions; subscriber-driven growth"),
    "Adobe":            ("Growth tech", [("0000796343", "Adobe")],
                         "Creative and document software, moved to subscriptions"),
    "Caterpillar":      ("Cyclical industrial", [("0000018230", "Caterpillar")],
                         "Construction and mining machinery; follows the investment cycle"),
    "Deere":            ("Cyclical industrial", [("0000315189", "Deere")],
                         "Farm and construction machinery; follows farm incomes and building"),
    "Cummins":          ("Cyclical industrial", [("0000026172", "Cummins")],
                         "Engines for trucks and machinery; follows truck and industrial cycles"),
    "PACCAR":           ("Cyclical industrial", [("0000075362", "PACCAR")],
                         "Heavy trucks (Kenworth, Peterbilt, DAF); orders swing with freight"),
    "Parker-Hannifin":  ("Cyclical industrial", [("0000076334", "Parker-Hannifin")],
                         "Motion-control parts for industrial and aerospace machinery"),
    "Procter & Gamble": ("FMCG", [("0000080424", "Procter & Gamble")],
                         "Household and personal-care brands; everyday purchases"),
    "Colgate-Palmolive": ("FMCG", [("0000021665", "Colgate-Palmolive")],
                          "Toothpaste, soap, pet food; everyday purchases"),
    "Clorox":           ("FMCG", [("0000021076", "Clorox")],
                         "Cleaning products, bags, cat litter; everyday purchases"),
    "Church & Dwight":  ("FMCG", [("0000313927", "Church & Dwight")],
                         "Baking soda, detergent, personal care; everyday purchases"),
    "General Mills":    ("FMCG", [("0000040704", "General Mills")],
                         "Packaged foods (cereal, snacks, pet food); everyday purchases"),
}


# Series that start later than the data (approved 17 Nov 2026):
# Best Buy moved its fiscal year-end from late February to late January in 2012
# (fiscal 2013 was a shortened transition year). Only the clean calendar is used.
SERIES_START = {"Best Buy": "2014Q1"}

# Session 4 hypothesis, fixed on 17 Nov 2026 BEFORE any Session 4 forecast was evaluated:
# which simple rule should win (lowest h = 1 MAPE) for each type.
HYPOTHESIS = {
    "Seasonal": "B1",             # same quarter last year: the seasonal pattern repeats
    "Growth tech": "B2",          # same quarter last year x growth: steady growth
    "Cyclical industrial": "B0",  # last quarter: the cycle moves on, year-old data is stale
    "FMCG": "B2",                 # Penelope: mild seasonality, steady low growth, price increases 2021-24
}


def company_type(company):
    return COMPANIES[company][0]
