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
    """rows: (the test and what we expect, the result, the edge case it covers)"""
    story.append(Paragraph(title, H3))
    table4(story, ("Test (and what we expect)", "Result (matched what we expected)", "Edge case"), rows, (3.1, 2.55, 0.85))


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
    story.append(Paragraph("Each test says what we expect to happen, and the result column says what did happen. All 32 tests passed, meaning every result matched the expectation. Run them with <font face='Courier'>python test_q1.py</font>. The last column says which edge case (P1 to P17) a test covers.", BODY))
    checks_table(story, "A. Ordinary situations", [
        ("<b>A1.</b> Empty station (0 people): the best action should be to wait, because a shuttle would cost 100 to carry nobody.",
         "The best action was to wait.", "P11"),
        ("<b>A2.</b> Dispatching with fewer than 15 people (4): the shuttle should take everybody, and we should pay only the 100.",
         "The cost was 100 and the next period had 1 to 5 people.", "P1"),
        ("<b>A3.</b> Does the 200-person limit matter? With a limit of 300 the best actions should not change.",
         "The actions were the same up to 150 people, and the costs differed by only 9e-11.", "P2"),
        ("<b>A4.</b> The chances of the next situation should add up to 100% everywhere.",
         "The largest error was 0, and a deliberately damaged table was caught by the same check.", "P3"),
        ("<b>A5.</b> A bigger shuttle should wait for more people before being sent.",
         "Shuttles of size 5, 15 and 30 dispatched at 5, 13 and 17 people.", "&ndash;"),
    ])
    checks_table(story, "B. Edges of the station", [
        ("<b>B1.</b> Exactly 15 people should empty the station, while 16 should leave one behind.",
         "The costs were 100 and 102.", "P1"),
        ("<b>B2.</b> Full station (200): we should dispatch, and waiting should not push it above 200.",
         "The best action was to dispatch, and waiting stayed at 200.", "P2"),
        ("<b>B3.</b> Nearly full (199 and 198): at 199 every arrival should be turned away, and at 198 only one should fit.",
         "From 199 the station always went to 200. From 198 it went to 199 with 20% chance and to 200 with 80%.", "P2"),
        ("<b>B4.</b> Off-by-one: there should be 201 situations (0 to 200), not 200.",
         "All tables are 201 wide and the graphs have 201 points.", "P4"),
    ])
    checks_table(story, "C. Arrivals", [
        ("<b>C1.</b> Nobody ever arrives and 10 are waiting (worked by hand): waiting forever costs 20 / (1 - 0.95) = 400 and dispatching once costs 100, "
         "so the cheapest should be 100.",
         "The program gave 100.000.", "&ndash;"),
        ("<b>C2.</b> Far more arrivals than the shuttle can carry (16 to 20 per period): the queue never empties, so each person left costs "
         "2 / (1 - 0.95) = 40 forever. A dispatch (100) should pay off from 100 / 40, so from 3 people. "
         "(Our first guess was 15; working it through showed the guess was wrong, not the program.)",
         "It dispatched for every number from 3 to 152.", "&ndash;"),
        ("<b>C2b.</b> Why does it stop at 152? Near the limit the queue hits 200 anyway, so a dispatch saves little, "
         "and the stopping point should move if the limit moves.",
         "The last dispatch was at 152 with a limit of 200, and at 252 with a limit of 300, so it is an effect of the cap.", "P2"),
    ])
    checks_table(story, "D. Extreme prices", [
        ("<b>D1a.</b> Free shuttle: we should dispatch whenever anyone is waiting.",
         "It dispatched in all 200 situations with someone waiting.", "&ndash;"),
        ("<b>D1b.</b> Free waiting: we should never dispatch, and the total cost should be 0.",
         "It never dispatched, and the cost was 0.", "&ndash;"),
        ("<b>D2.</b> Enormous shuttle cost (1,000,000): we should never dispatch, even with a full station.",
         "It never dispatched.", "&ndash;"),
    ])
    checks_table(story, "E. Time and discounting", [
        ("<b>E1.</b> Very short horizon (worked by hand): with 1 period left the cost should be 2 times the number of people, "
         "and with 2 periods left 3.9 times the people plus 5.7.",
         "Both matched the program exactly.", "P6"),
        ("<b>E2.</b> Is 500 periods long enough? As the horizon grows from 100 to 200 to 500 periods, the answer should settle down.",
         "The difference from the no-end-date answer was 4.0, then 0.024, then 0.000000003.", "P6"),
    ])
    checks_table(story, "F. Checks on each method", [
        ("<b>F1.</b> Enumeration starting point: nothing should be left to pay after the last period, and more periods left should never cost less.",
         "The value after period 500 is 0 everywhere, and costs only grew with more periods.", "P6"),
        ("<b>F2.</b> Value iteration should reach the same answer from different starting guesses.",
         "Starting guesses of all 0, all 1000 and random numbers agreed to 4e-9.", "P7"),
        ("<b>F3.</b> Policy iteration should improve the plan every round, and when all actions cost the same it should not loop forever.",
         "It took 6 rounds and the cost never went up. A problem where every action ties stopped after 1 round.", "P8, P9"),
    ])
    checks_table(story, "G. Overall checks", [
        ("<b>G1.</b> All three methods should agree on the costs and on the plan.",
         "The biggest gap was 3e-9, and the plan was identical.", "all"),
        ("<b>G2.</b> Policy iteration started from a bad plan (&ldquo;always dispatch&rdquo;) should still find the same best plan.",
         "It did.", "P8"),
        ("<b>G3.</b> Simulating the station 3000 times should give an average cost that matches the calculated cost.",
         "The simulation gave 575.8 (plus or minus 1.7) against a calculated 576.7.", "&ndash;"),
    ])
    checks_table(story, "H. The safety checks themselves", [
        ("<b>H1.</b> Nonsense numbers (no arrivals, negative arrivals, sizes or costs, a limit of 0) should each be rejected with a clear message.",
         "All 6 were rejected.", "P12"),
        ("<b>H1b.</b> Tables of the wrong size, or whose chances do not add to 100%, should be rejected.",
         "Both were stopped.", "P12"),
        ("<b>H2.</b> A discount of 1.0, 1.5 or -0.1 should be rejected, while a discount of 0.0 should be allowed.",
         "The first three were rejected and 0.0 was accepted.", "P13"),
        ("<b>H3.</b> Value iteration given a target it can never meet should stop with an error after the maximum number of rounds.",
         "It stopped with an error (50 rounds were allowed).", "P14"),
        ("<b>H4.</b> Policy iteration allowed only 1 round (it needs 6) should stop with an error.",
         "It stopped with an error.", "P15"),
        ("<b>H5.</b> Enumeration with 0, -3 or 2.5 periods should be rejected.",
         "All three were rejected.", "P16"),
        ("<b>H6.</b> A NaN or infinite cost should make all three methods stop with an error.",
         "All three methods stopped. (An infinite cost on an action nobody uses is rightly not an error.)", "P17"),
        ("<b>H7.</b> Waiting with 10 people: we should pay 20, and next period should have 11 to 15 people.",
         "The cost was 20, with a 20% chance of each of 11 to 15 people.", "P5"),
        ("<b>H8.</b> Exact equation solving should agree with a slow repeat-until-settled calculation.",
         "The largest gap was 9e-13.", "P10"),
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
