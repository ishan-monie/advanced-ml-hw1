"""Builds report.pdf. The report is built one section at a time; add new sections to SECTIONS below.
Run: python make_report.py"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted,
                                PageBreak, Image, KeepTogether)
from PIL import Image as PILImage

HERE = os.path.dirname(os.path.abspath(__file__))
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontSize=11, leading=15, spaceAfter=8)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontSize=20, spaceAfter=14)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceBefore=14, spaceAfter=8)
CODE = ParagraphStyle("code", parent=ss["Code"], fontSize=9.5, leading=12, backColor=colors.HexColor("#f2f2f2"),
                      borderPadding=6, spaceAfter=10, leftIndent=6)
LINK = ParagraphStyle("link", parent=BODY, textColor=colors.HexColor("#1a55b0"))

GITHUB_USER = "ishan-monie"
REPO_NAME = "advanced-ml-hw1"
REPO_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}"


def summary_section(story):
    story.append(Paragraph("Assignment 1 &ndash; Summary", TITLE))

    story.append(Paragraph("GitHub repository", H1))
    t = Table([["GitHub username", GITHUB_USER], ["Repository name", REPO_NAME],
               ["Link to the code", Paragraph(f'<link href="{REPO_URL}">{REPO_URL}</link>', LINK)]],
              colWidths=[1.7 * inch, 4.9 * inch])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 11),
                           ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef2f7")),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t)
    story.append(Spacer(1, 8))
    story.append(Paragraph("All of the code used for this assignment is in this repository.", BODY))

    story.append(Paragraph("How to run the code", H1))
    story.append(Paragraph("<b>Step 1.</b> Download the repository (green <i>Code</i> button, then <i>Download ZIP</i>) "
                           "or clone it, and install the Python packages:", BODY))
    story.append(Preformatted("pip install numpy matplotlib", CODE))
    story.append(Paragraph("<b>Step 2.</b> Shuttle problem. Open a terminal in the <font face='Courier'>q1_shuttle</font> folder and run:", BODY))
    story.append(Preformatted("python run_q1.py     # solves the problem 3 ways and saves the 3 plots in figures/\n"
                              "python test_q1.py    # runs the 32 edge-case and correctness checks (PASS/FAIL for each)", CODE))
    story.append(Paragraph("The README in the repository repeats these instructions and explains what each file does. "
                           "Instructions for the policy gradient questions will be added to the same README.", BODY))


CAPTION = ParagraphStyle("caption", parent=BODY, fontSize=9.5, leading=12, alignment=1,
                         textColor=colors.HexColor("#444444"), spaceAfter=12)
FIG_DIR = os.path.join(HERE, "..", "q1_shuttle", "figures")


def figure(story, filename, caption, width=3.2 * inch):
    path = os.path.join(FIG_DIR, filename)
    w, h = PILImage.open(path).size
    story.append(KeepTogether([Image(path, width=width, height=width * h / w), Spacer(1, 3), Paragraph(caption, CAPTION)]))


def shuttle_section(story):
    story.append(PageBreak())
    story.append(Paragraph("Question 2: Value and Policy Iteration (the shuttle problem)", TITLE))
    story.append(Paragraph("Each period we choose whether to dispatch a shuttle (cost 100, takes up to 15 people) or wait "
                           "(every person left waiting costs 2). Between 1 and 5 new people arrive each period. "
                           "We solve this three ways. A lower cost is better in every graph.", BODY))

    story.append(Paragraph("Assumptions", H1))
    story.append(Paragraph("The problem leaves some details open, so these are the choices we made:", BODY))
    assumptions = [
        "<b>What we track.</b> The only thing that matters when deciding is how many people are waiting at the start of a period "
        "(0 to 200). We do not track the time of day or what happened before.",
        "<b>Costs, not rewards.</b> Everything is a cost, so we look for the <i>lowest</i> total cost. The &ldquo;value&rdquo; in the graphs is the "
        "expected total cost, so lower is better.",
        "<b>Order of events in a period.</b> We first decide (wait or dispatch), then pay, then the new people arrive.",
        "<b>Dispatching.</b> The shuttle is always available. It takes up to 15 people, or everybody if fewer than 15 are waiting. "
        "We allow dispatching with an empty station, even though it is wasteful.",
        "<b>The waiting cost.</b> The cost of 2 per person is charged on the people <i>left</i> after the shuttle has gone "
        "(everyone, if we wait).",
        "<b>Full station.</b> The station holds at most 200 people. If more would arrive, the extra people are turned away "
        "(they are lost, they do not wait). We checked that this choice does not change the answer.",
        "<b>Arrivals.</b> Each period 1, 2, 3, 4 or 5 people arrive, each with chance 1/5, independent of the past.",
        "<b>Discounting.</b> A cost one period later counts 0.95 as much. The cost in the first period is not discounted.",
        "<b>The end of the 500 periods (enumeration).</b> After the last period nothing more is paid, so people still "
        "waiting at that point cost nothing extra.",
        "<b>The plan.</b> The best action depends only on the number of people waiting, and it is the same in every "
        "period (this is what value iteration and policy iteration find).",
        "<b>Stopping and ties.</b> Value iteration stops when no value changes by more than 1e-10. Policy iteration starts from "
        "&ldquo;never dispatch&rdquo; and keeps its current action when two actions cost exactly the same.",
    ]
    for a in assumptions:
        story.append(Paragraph(a, ParagraphStyle("li", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=4), bulletText="•"))
    story.append(Spacer(1, 6))

    story.append(Paragraph("(a) Enumeration, 500 periods", H1))
    story.append(Paragraph("We start at the end of the 500 periods, where nothing is left to pay, and work backwards, "
                           "keeping the cheaper of the two actions at every step. The graph shows the best expected total "
                           "cost at time 0 for each number of people waiting.", BODY))
    figure(story, "a_enumeration.png", "Figure 1. Enumeration: best expected cost at time 0 versus people waiting.")

    story.append(Paragraph("(b) Value iteration, no end date", H1))
    story.append(Paragraph("The same step is repeated with no end date until the numbers stop changing (519 rounds). "
                           "The curve is the same as in (a), which is expected: after 500 periods the far future is "
                           "discounted to almost nothing, and the two differ by about 3 in a billion.", BODY))
    figure(story, "b_value_iteration.png", "Figure 2. Value iteration: best expected cost versus people waiting.")
    figure(story, "ab_overlay.png", "Figure 3. Figures 1 and 2 on the same axes. The two curves lie on top of each other.", width=3.2 * inch)

    story.append(Paragraph("(c) Policy iteration", H1))
    story.append(Paragraph("Here we improve a plan instead of the costs. Starting from &ldquo;never dispatch&rdquo;, we measure how good "
                           "the plan is, change it wherever another action is cheaper, and repeat. It settled after 6 rounds. "
                           "The best plan is to wait while fewer than 13 people are waiting and to dispatch at 13 or more.", BODY))
    figure(story, "c_policy_iteration.png", "Figure 4. Policy iteration: best action versus people waiting.")


SMALL = ParagraphStyle("small", parent=BODY, fontSize=9, leading=11, spaceAfter=0)
SMALLB = ParagraphStyle("smallb", parent=SMALL, fontName="Helvetica-Bold")
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=12, spaceAfter=5, keepWithNext=1)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontSize=10.5, spaceBefore=9, spaceAfter=4, keepWithNext=1)


def table4(story, headers, rows, widths):
    data = [[Paragraph(h, SMALLB) for h in headers]] + [[Paragraph(c, SMALL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(t)


def checks_table(story, title, rows):
    """rows: (case, what we expected, what we found, protection number)"""
    story.append(Paragraph(title, H3))
    table4(story, ("Test", "What we expected", "What we found", "Edge case"), rows, (1.9, 1.85, 1.85, 0.9))


def edge_cases_section(story):
    story.append(PageBreak())
    story.append(Paragraph("Edge cases", H1))
    story.append(Paragraph("The unusual situations we considered, and how the code handles each one (P1 to P17).", BODY))
    story.append(Paragraph("Built into the calculation", H3))
    table4(story, ("#", "Unusual situation", "How the code handles it"), [
        ("P1", "Shuttle leaves with fewer than 15 people", "It takes everybody; the number left is never below 0 (<font face='Courier'>max(people - 15, 0)</font>)."),
        ("P2", "Arrivals would push the station past 200", "The station stops at 200 and the extra people are turned away (<font face='Courier'>min(..., 200)</font>)."),
        ("P3", "Chances of the next situation not adding up to 100%", "Each possible arrival number gets exactly 1 / (number of options), so they always add up to 100%."),
        ("P4", "Off-by-one: 0 to 200 is 201 situations, not 200", "Every table is sized 200 + 1."),
        ("P5", "Waiting should not remove anybody", "When we wait, everyone stays; we pay 2 for each."),
        ("P6", "Enumeration needs a starting point", "After the last period the value is 0, then it works backwards from period 499 to 0."),
        ("P7", "Value iteration stopping too early or never", "It stops only when no value changes by more than 1e-10, not after a fixed number of rounds."),
        ("P8", "Policy iteration looping on ties or rounding noise", "It only changes an action if the new one is better by more than 1e-9; otherwise it keeps the current action."),
        ("P9", "Value iteration flip-flopping on exact ties", "Dispatch is chosen only if it is strictly cheaper; a tie means wait."),
        ("P10", "Inaccurate evaluation of a plan", "Policy iteration solves the exact equations instead of approximating."),
        ("P11", "Dispatching an empty station", "Allowed, with no special rule. It is just never the best action."),
    ], (0.45, 2.5, 3.6))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Safety checks added to stop bad inputs and runaway calculations", H3))
    table4(story, ("#", "Unusual situation", "How the code handles it"), [
        ("P12", "Nonsense numbers: no arrivals, negative arrivals, sizes or costs, a station limit of 0, tables that do not match the problem",
         "It stops at the start with a clear message saying what is wrong."),
        ("P13", "A discount of 1 or more (or below 0)", "Rejected. With a discount of 1 the &ldquo;forever&rdquo; problem has no finite answer."),
        ("P14", "Value iteration never settles", "A maximum number of rounds; if it is exceeded the program stops with an error instead of running forever."),
        ("P15", "Policy iteration never settles", "The same maximum for policy iteration."),
        ("P16", "A strange horizon for enumeration (0, negative or a fraction of a period)", "Rejected: it needs a whole number of periods, at least 1."),
        ("P17", "A value turns into NaN or infinity", "The calculation stops with an error, so bad numbers cannot spread quietly."),
    ], (0.45, 3.0, 3.1))
    story.append(Spacer(1, 4))
    story.append(Paragraph("We did not guard against things like typing text instead of a number: that is careless typing, not part of this problem, "
                           "and the assignment fixes all the numbers.", SMALL))

    story.append(PageBreak())
    story.append(Paragraph("Tests", H1))
    story.append(Paragraph("Checks that the solution gives the right answers, including when the problem is changed in ways where we can predict the result. All 32 pass; run them with <font face='Courier'>python test_q1.py</font>. The last column says which edge case (P1 to P17) a test covers.", BODY))
    checks_table(story, "A. Ordinary situations", [
        ("A1. Empty station (0 people)", "Wait: a shuttle would cost 100 to carry nobody.", "Best action is to wait.", "P11"),
        ("A2. Dispatching with fewer than 15 people (4)", "The shuttle takes everybody; we pay only the 100.", "Cost 100; next period has 1 to 5 people.", "P1"),
        ("A3. Does the 200-person limit matter? Try 300.", "No change to the best actions.", "Same actions up to 150 people; costs differ by 9e-11.", "P2"),
        ("A4. Chances of the next situation add up to 100%", "Exactly 100% in every situation.", "Largest error 0. A deliberately damaged table is caught by the same check.", "P3"),
        ("A5. A bigger shuttle", "Wait for more people before sending it.", "Shuttle of 5 / 15 / 30 dispatches at 5 / 13 / 17 people.", "&ndash;"),
    ])
    checks_table(story, "B. Edges of the station", [
        ("B1. Exactly 15 people, then 16", "15 empties the station; 16 leaves one behind.", "Costs 100 and 102.", "P1"),
        ("B2. Full station (200)", "Dispatch, and waiting cannot push it above 200.", "Best action is to dispatch; waiting stays at 200.", "P2"),
        ("B3. Nearly full (199 and 198)", "At 199 every arrival is turned away. At 198 only one fits.", "From 199: always 200. From 198: 20% to 199, 80% to 200.", "P2"),
        ("B4. Off-by-one", "201 situations (0 to 200), not 200.", "All tables are 201 wide and the graphs have 201 points.", "P4"),
    ])
    checks_table(story, "C. Arrivals", [
        ("C1. Nobody ever arrives, 10 waiting (worked by hand)", "Waiting forever costs 20/(1-0.95) = 400; dispatching once costs 100. Cheapest: 100.", "The program gives 100.000.", "&ndash;"),
        ("C2. Far more arrivals than the shuttle can carry (16 to 20 per period)",
         "The queue never empties, so each person left costs 2/(1-0.95) = 40 forever. A dispatch (100) pays off from 100/40, so from 3 people. "
         "(Our first guess here was 15; working it through showed the guess was wrong, not the program.)",
         "Dispatches for every number from 3 to 152.", "&ndash;"),
        ("C2b. Why does it stop at 152?", "Near the limit the queue hits 200 anyway, so a dispatch saves little. The point should move if the limit moves.",
         "Last dispatch at 152 (limit 200) and 252 (limit 300): an effect of the cap.", "P2"),
    ])
    checks_table(story, "D. Extreme prices", [
        ("D1a. Free shuttle", "Dispatch whenever anyone is waiting.", "Dispatches in 200 of 200 situations.", "&ndash;"),
        ("D1b. Free waiting", "Never dispatch; total cost 0.", "Never dispatches; cost 0.", "&ndash;"),
        ("D2. Enormous shuttle cost (1,000,000)", "Never dispatch, even with a full station.", "Never dispatches.", "&ndash;"),
    ])
    checks_table(story, "E. Time and discounting", [
        ("E1. Very short horizon (worked by hand)", "1 period left costs 2 x people. 2 periods left costs 3.9 x people + 5.7.", "Both match the program exactly.", "P6"),
        ("E2. Is 500 periods long enough? Compare 100, 200 and 500.", "The answer settles down as the horizon grows.",
         "Difference from the no-end-date answer: 4.0 / 0.024 / 0.000000003.", "P6"),
    ])
    checks_table(story, "F. Checks on each method", [
        ("F1. Enumeration: the starting point", "Nothing left to pay after the last period; more periods left never costs less.", "Value after period 500 is 0 everywhere; costs only grow with more periods.", "P6"),
        ("F2. Value iteration: different starting guesses", "Same answer from any guess.", "All 0, all 1000 and random guesses agree to 4e-9.", "P7"),
        ("F3. Policy iteration: improving and ties", "Each round improves the plan. When all actions cost the same, it must not loop forever.", "6 rounds, costs never went up. The all-ties problem stopped after 1 round.", "P8, P9"),
    ])
    checks_table(story, "G. Overall checks", [
        ("G1. All three methods agree", "Same costs and same plan.", "Biggest gap 3e-9; identical plan.", "all"),
        ("G2. Policy iteration from a bad start (&ldquo;always dispatch&rdquo;)", "Still finds the same best plan.", "It does.", "P8"),
        ("G3. Simulate the station 3000 times", "The average cost matches the calculated cost.", "Simulated 575.8 (&plusmn;1.7) against calculated 576.7.", "&ndash;"),
    ])
    checks_table(story, "H. The safety checks themselves", [
        ("H1. Nonsense numbers (no arrivals, negative arrivals, sizes or costs, limit 0)", "Each one is rejected with a clear message.", "6 of 6 rejected.", "P12"),
        ("H1b. Tables of the wrong size, or whose chances do not add to 100%", "Rejected.", "Both stopped.", "P12"),
        ("H2. Discount of 1.0, 1.5 or -0.1; and 0.0", "The first three are rejected; 0.0 is allowed.", "As expected.", "P13"),
        ("H3. Value iteration with a target it can never meet", "Stops with an error after the maximum rounds.", "It does (50 rounds allowed).", "P14"),
        ("H4. Policy iteration allowed only 1 round (it needs 6)", "Stops with an error.", "It does.", "P15"),
        ("H5. Enumeration with 0, -3 or 2.5 periods", "Rejected.", "All three rejected.", "P16"),
        ("H6. A NaN or infinite cost", "All three methods stop with an error.", "They do. (An infinite cost on an action nobody uses is, rightly, not an error.)", "P17"),
        ("H7. Waiting with 10 people", "We pay 20; next period has 11 to 15 people.", "Cost 20; chances 20% each for 11 to 15.", "P5"),
        ("H8. Exact equation solving vs a slow repeat-until-settled calculation", "Same costs.", "Largest gap 9e-13.", "P10"),
    ])


def three_months_section(story):
    block = []
    story_real, story = story, block
    story.append(Paragraph("With 3 more months", H1))
    story.append(Paragraph("<b>Make the model more realistic.</b> We assumed 1 to 5 people arrive with equal chance. With real arrival data we would "
                           "check whether &ldquo;dispatch at 13&rdquo; still holds, and add rush hours, travel time, several shuttles and people who give up "
                           "waiting. We would also see how far the answer moves if the costs of 100 and 2 are really 50% to 150% of what we assumed, and ask "
                           "someone who runs a station whether our assumptions (such as when the waiting cost is charged) match reality.", BODY))
    story.append(Paragraph("<b>Test the solution harder.</b> We would generate hundreds of random shuttle problems and check that all three methods agree "
                           "on every one, compare against a brute-force search over every plan on small versions, and simulate rival rules (&ldquo;dispatch "
                           "at 10&rdquo;, &ldquo;at 15&rdquo;, &ldquo;at 20&rdquo;) for a long time to show with confidence ranges that 13 really is cheapest.", BODY))
    story.append(Paragraph("<b>Check it a different way.</b> We would let a learning method, like the policy gradient methods in the next questions, "
                           "learn from simulated stations without being told the rules, and see whether it also finds about 13. We would also check that "
                           "the program stays fast and accurate on a much bigger station.", BODY))
    story_real.append(KeepTogether(block))


SECTIONS = [summary_section, shuttle_section, edge_cases_section, three_months_section]

story = []
for add in SECTIONS:
    add(story)
SimpleDocTemplate(os.path.join(HERE, "report.pdf"), pagesize=letter, leftMargin=inch, rightMargin=inch,
                  topMargin=0.9 * inch, bottomMargin=0.9 * inch, title="Advanced ML Assignment 1").build(story)
print("wrote report.pdf")
