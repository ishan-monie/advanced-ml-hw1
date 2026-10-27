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
                              "python test_q1.py    # runs the 13 edge-case and correctness checks (PASS/FAIL for each)", CODE))
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


SECTIONS = [summary_section, shuttle_section]

story = []
for add in SECTIONS:
    add(story)
SimpleDocTemplate(os.path.join(HERE, "report.pdf"), pagesize=letter, leftMargin=inch, rightMargin=inch,
                  topMargin=0.9 * inch, bottomMargin=0.9 * inch, title="Advanced ML Assignment 1").build(story)
print("wrote report.pdf")
