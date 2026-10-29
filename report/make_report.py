"""Builds report.pdf. The report is built one section at a time; add new sections to SECTIONS below.
Run: python make_report.py"""
import os
import numpy as np
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


PG_FIG_DIR = os.path.join(HERE, "..", "q3_policy_gradient", "figures")


def figure_from(story, folder, filename, caption, width=3.7 * inch):
    path = os.path.join(folder, filename)
    w, h = PILImage.open(path).size
    story.append(KeepTogether([Image(path, width=width, height=width * h / w), Spacer(1, 3), Paragraph(caption, CAPTION)]))


def policy_gradient_section(story):
    story.append(PageBreak())
    story.append(Paragraph("Question 3: Policy Gradient, Part 1", TITLE))
    story.append(Paragraph("We teach a small neural network to play a game by trial and error. It outputs a probability for each action and plays "
                           "a whole game choosing actions at random from those probabilities. Afterwards, actions that were followed by lots of "
                           "reward are made more likely, and actions followed by little reward are made less likely. This is the basic "
                           "version, with no baseline (the next part adds one).", BODY))

    story.append(Paragraph("Assumptions", H1))
    story.append(Paragraph("These are the choices we made for part (a), CartPole-v1:", BODY))
    assumptions = [
        "<b>One game is one episode.</b> CartPole ends when the pole tips too far, the cart leaves the track, or after 500 steps. "
        "Every step the pole stays up gives a reward of 1, so the best possible score is 500.",
        "<b>The network.</b> It sees 4 numbers (cart position, cart speed, pole angle, pole speed), has one hidden layer of 128 units, "
        "and outputs the probability of pushing left or right.",
        "<b>Actions are random.</b> Both during training and during the final 500 games, each push is chosen at random using the network&rsquo;s "
        "probabilities, rather than always taking the most likely one.",
        "<b>Credit for each push.</b> Each push is credited with the reward from that step onwards, discounted by 0.95 per step "
        "(as the assignment says), not with the whole game&rsquo;s total.",
        "<b>Learning.</b> One update after every game, using Adam with a learning rate of 0.01 (as the assignment says). "
        "There is no baseline and the credits are not rescaled, because this is the basic version.",
        "<b>Training length and luck.</b> We train for 1500 games. Results depend on the random starting point, so the main model "
        "uses seed 0, chosen in advance, and two more seeds are shown to see how much luck matters.",
        "<b>Final test.</b> After training the network stops learning. We play 500 new games, with a different random seed, and plot the scores.",
        "<b>Moving average.</b> The average of the last 100 games (fewer at the very start).",
        "<b>The 500-step limit</b> is treated like any other end of the game.",
        "<b>Bad numbers.</b> If an update ever produced NaN or infinity, it would be skipped instead of damaging the network. "
        "This never happened in our runs.",
    ]
    for a in assumptions:
        story.append(Paragraph(a, ParagraphStyle("li3", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=4), bulletText="•"))

    story.append(PageBreak())
    story.append(Paragraph("(a) CartPole-v1", H1))
    story.append(Paragraph("We tried learning rates of 0.01, 0.003 and 0.001 on three seeds each. The averages of the last 100 training games were "
                           "285, 249 and 234, but they varied a lot between seeds (for example 500 against 163 at the same rate), so no rate "
                           "was clearly better and we kept the assignment&rsquo;s 0.01.", BODY))
    figure_from(story, PG_FIG_DIR, "cartpole_training.png",
                "Figure 5. Reward of every training game (grey) and the moving average of the last 100 games (red).", width=4.1 * inch)
    story.append(Paragraph("The network clearly learns: early games last about 20 steps and later ones often reach 200 or more. But it is "
                           "unsteady. The average rises, falls back, and rises again, and the main model finished at an average of about 164 "
                           "over its last 100 training games. This noise is typical of the basic method and is what a baseline is meant to reduce.", BODY))
    figure_from(story, PG_FIG_DIR, "cartpole_histogram.png",
                "Figure 6. Scores of 500 games played by the trained main model (mean 155.4, standard deviation 12.8).", width=3.6 * inch)
    figure_from(story, PG_FIG_DIR, "cartpole_seeds.png",
                "Figure 7. The same training with three different random seeds. Seed 1 reaches the 500 limit and stays there; seeds 0 and 2 do not.", width=4.1 * inch)
    story.append(Paragraph("On the 500 test games, the main model (seed 0) scored a mean of 155.4 with a standard deviation of 12.8. "
                           "The other seeds scored a mean of 500.0 (seed 1, with no variation) and 277.5 (seed 2). Where the model ends up "
                           "depends a lot on the seed, so a single run should not be over-interpreted.", BODY))


def cartpole_edge_cases_section(story):
    story.append(PageBreak())
    story.append(Paragraph("CartPole: Edge cases", H1))
    story.append(Paragraph("The unusual situations we considered for part (a), and how the code handles each one (R1 to R9).", BODY))
    table4(story, ("#", "Unusual situation", "How the code handles it"), [
        ("R1", "A reward crediting pushes that came before it", "Credits are worked out backwards with a running total, so each push only gets the reward from its own step onwards."),
        ("R2", "Bad settings: fewer than 1 game, a learning rate of 0 or below, a discount below 0 or of 1 or more",
         "Rejected at the start with a clear message. With a discount of 1 or more the credits could grow without limit."),
        ("R3", "A NaN or infinite number in the learning step", "That update is skipped and counted, so the network cannot be damaged. A warning is printed if it ever happens (it did not in our runs)."),
        ("R4", "An empty or very short game", "An empty game is skipped. A one-step game still gives a valid update."),
        ("R5", "Taking the log of a probability that is 0, or huge numbers", "Probabilities are built from the raw scores, so they are never exactly 0 and stay finite even for extreme networks or inputs."),
        ("R6", "How a game ends", "The pole falling and the 500-step limit both simply end the game. We use CartPole-v1, so a score can never be above 500."),
        ("R7", "Test games changing the network by accident", "Test games switch off learning information, and the network is not updated during them."),
        ("R8", "Results that cannot be repeated", "Random seeds are fixed, and the 500 test games use a different seed from training."),
        ("R9", "The moving average at the start, before 100 games exist", "It averages over however many games exist so far, instead of failing or counting empty games as 0."),
    ], (0.45, 2.45, 3.6))

    story.append(PageBreak())
    story.append(Paragraph("CartPole: Tests", H1))
    story.append(Paragraph("Each test says what we expect to happen, and the result column says what did happen. All 24 tests passed, meaning "
                           "every result matched the expectation. Run them with <font face='Courier'>python test_cartpole.py</font>. "
                           "The last column says which edge case (R1 to R9) a test covers.", BODY))
    checks_table(story, "A. Credits and game rules", [
        ("<b>A1.</b> Worked by hand: rewards 1, 1, 1 with discount 0.5 should give credits 1.75, 1.5 and 1.0, and a reward should never credit earlier pushes.",
         "The credits were 1.75, 1.5 and 1.0, and a lone reward of 5 credited only its own step.", "R1"),
        ("<b>A2.</b> A one-step game (the pole falls straight away) should still give a valid update.",
         "The update worked and the network changed.", "R4"),
        ("<b>A3.</b> A discount of 0 should give each push only its own reward, 0.99 should work, and 1.0 or -0.1 should be rejected.",
         "Discounts of 0 and 0.99 worked; 1.0 and -0.1 were rejected.", "R2"),
        ("<b>A4.</b> In the longest game (500 steps) no credit should exceed 1 / (1 - 0.95) = 20.",
         "The largest credit was 20.000.", "&ndash;"),
        ("<b>A5.</b> We should be using CartPole-v1 with a 500-step limit (not v0, which stops at 200).",
         "It is CartPole-v1, with a limit of 500.", "R6"),
        ("<b>A6.</b> Every step should give a reward of 1, so a game&rsquo;s score equals its length.",
         "A sample game of 36 steps scored 36.", "&ndash;"),
    ])
    checks_table(story, "B. The network", [
        ("<b>B1.</b> Before any training, both actions should be about equally likely.",
         "The average chance of pushing left was 0.527.", "&ndash;"),
        ("<b>B2.</b> The two probabilities should add up to 100% for every state, with no NaN.",
         "The largest error was about 1e-7.", "R5"),
        ("<b>B3.</b> An extreme network (weights multiplied by 10,000) and extreme inputs (millions) should still give finite log-probabilities.",
         "They did.", "R5"),
    ])
    checks_table(story, "C. The training loop", [
        ("<b>C1.</b> A NaN reward should make the update skip, leaving the network untouched.",
         "The update was skipped and the weights were identical.", "R3"),
        ("<b>C2.</b> An empty game should be skipped instead of crashing.", "It was skipped.", "R4"),
        ("<b>C3.</b> Bad settings (0 games, a learning rate of 0 or below) should be rejected with a message.",
         "All three were rejected.", "R2"),
        ("<b>C4.</b> The same seed should give an identical training curve, and a different seed a different one.",
         "Two runs with seed 3 were identical, and seed 4 differed.", "R8"),
        ("<b>C5.</b> Test games should never change the network.",
         "The weights were identical before and after, and no learning information was kept.", "R7"),
        ("<b>C6.</b> A made-up problem with a known answer (one state where action 1 always pays 1 and action 0 pays 0): "
         "our learning signal should match the exact gradient of the expected reward.",
         "The direction matched (1.0000) and the size was within 0.9%.", "&ndash;"),
        ("<b>C6b.</b> The previous check should really be able to fail: a learner with the sign flipped, or one that ignores the reward, should be caught.",
         "Both were caught (a direction match of -1.00 instead of +1.00).", "&ndash;"),
        ("<b>C7.</b> In the same made-up problem, training should make the better action much more likely.",
         "Its chance went from 0.47 to 1.00.", "&ndash;"),
    ])
    checks_table(story, "D. Does it behave as predicted?", [
        ("<b>D1.</b> The trained network should be far better than an untrained one.",
         "155.4 against 22.0 on average (a statistical test gave t = 176).", "&ndash;"),
        ("<b>D2.</b> With a tiny learning rate (1e-6) it should hardly learn and stay near random play (about 22).",
         "The average of the last 100 games was 20.2.", "&ndash;"),
        ("<b>D3.</b> Taking away the randomness (always the most likely action) should do about as well or better.",
         "155.9 against 155.4.", "&ndash;"),
        ("<b>D4.</b> The last 100 training games should score much higher than the first 100.",
         "The average was 31.4 at the start and 163.5 at the end.", "&ndash;"),
    ])
    checks_table(story, "E. The 500 test games and the graphs", [
        ("<b>E1.</b> There should be exactly 500 test games, each scoring between 1 and 500.",
         "500 games, with scores from 128 to 192, a mean of 155.4 and a standard deviation of 12.8.", "R6"),
        ("<b>E2.</b> The moving average, worked by hand: [1, 2, 3, 4] with a window of 2 should give 1, 1.5, 2.5, 3.5. With a window of 100 the first value "
         "should be the first game, game 100 the mean of games 1 to 100, and game 101 should drop game 1.",
         "All of these matched.", "R9"),
        ("<b>E3.</b> The saved model should reproduce exactly the 500 test scores behind the graphs in this report, and retraining from the same seed should repeat the start of the saved training curve.",
         "The 500 test scores were identical, and so were the first 200 training games.", "R8"),
    ])


def pong_section(story):
    story.append(PageBreak())
    story.append(Paragraph("(b) Pong-v5", H1))
    story.append(Paragraph("The same method as for CartPole, now on the Atari game Pong. The network sees the screen, outputs the chance of moving "
                           "the paddle up, and plays whole games. These are the choices we made for part (b):", BODY))
    assumptions = [
        "<b>The game.</b> We play the right-hand paddle against the computer, using ALE/Pong-v5. A game ends when one side reaches 21 points. "
        "Each point we win gives +1 and each point we lose gives &minus;1, so a game scores between &minus;21 and +21. Random play scores about &minus;20.",
        "<b>The game is partly random.</b> In Pong-v5 each push is held for 4 screen frames, and with a 25% chance the game repeats our previous "
        "push instead of the new one. So even identical play gives different results, and scores vary from game to game.",
        "<b>What the network sees.</b> Each screen is cleaned with the assignment&rsquo;s <font face='Courier'>preprocess</font> function into an "
        "80 x 80 grid of 0s and 1s (paddles and ball are 1). The network is shown the <i>difference</i> between this frame and the previous one, "
        "so it can tell which way things are moving. The first frame of a game has no previous frame, so its difference is all zeros. "
        "<font face='Courier'>preprocess</font> changes the picture it is given, so we always give it a copy.",
        "<b>The network.</b> 6,400 inputs, one hidden layer of 256 units, and one output: the chance of pushing up.",
        "<b>Actions.</b> Of the 6 actions in the game, we use only RIGHT (action 2, paddle up) and LEFT (action 3, paddle down), as the assignment says.",
        "<b>Actions are random.</b> Both in training and in the final 500 games, each push is chosen at random using the network&rsquo;s probability.",
        "<b>Credit for each push.</b> Discount 0.99, as the assignment says. The running total restarts at every point scored, so a push is only "
        "credited for the point it was part of (each point is treated as its own small game). This is a common choice for Pong, but it is "
        "our choice, not something the assignment states.",
        "<b>Learning.</b> Adam, with the loss averaged over 4 games per update (one update per game would be very noisy for a network this big). "
        "No baseline and no rescaling of the credits, because this is the basic version.",
        "<b>Learning rate.</b> The assignment gives 0.01 but allows a different value if training is more stable. We tried both 0.01 and "
        "0.001. Our rule, set before seeing the results: the main model is the one with the higher average over its last 100 "
        "training games, and both are shown. A second run at 0.001 with a different random seed shows how much luck matters.",
        "<b>Training length.</b> A Pong game takes one to several seconds of computer time (longer as the player improves), so runs are given a fixed "
        "time limit instead of a fixed number of games. The two 0.001 runs had 6 hours on a GPU server (one card) and played 6,993 (main) and 6,534 games. "
        "The 0.01 run had 4 hours on a laptop (9,486 games). An earlier pair of 0.001 runs on a laptop was cut short when the laptop went to sleep "
        "(about 3,300 and 2,900 games) and was replaced by these longer runs.",
        "<b>Final test.</b> After training the network stops learning. We play 500 new games, with a different random seed, and plot the scores.",
        "<b>Bad numbers.</b> An update that would produce NaN or infinity is skipped, and skipped updates are counted in the saved settings.",
    ]
    for a in assumptions:
        story.append(Paragraph(a, ParagraphStyle("li4", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=4), bulletText="•"))


def pong_edge_cases_section(story):
    story.append(PageBreak())
    story.append(Paragraph("Pong: Edge cases", H1))
    story.append(Paragraph("The unusual situations we considered for part (b), and how the code handles each one (S1 to S10).", BODY))
    table4(story, ("#", "Unusual situation", "How the code handles it"), [
        ("S1", "A lost point blaming pushes from an earlier point", "The running total of credit restarts at every point scored, so a push is only credited for its own point."),
        ("S2", "Bad settings: batch below 1, a learning rate of 0 or below, a discount below 0 or of 1 or more", "Rejected at the start with a clear message."),
        ("S3", "The very first screen of a game, which has no screen before it", "The network is shown all zeros for that step instead of failing."),
        ("S4", "The cleaning function changing the game&rsquo;s own picture", "The function edits what it is given, so we always give it a copy."),
        ("S5", "Using the wrong buttons", "Of the 6 actions we use only 2 (up) and 3 (down), and a test confirms what those buttons are called in the game."),
        ("S6", "A NaN or infinite number in the learning step, or a batch with no games", "That update is skipped and counted, so the network cannot be damaged."),
        ("S7", "Taking the log of a probability that is 0, or huge numbers", "Probabilities are built from the raw score, so they stay finite even for extreme networks or inputs."),
        ("S8", "Running out of time or games in the middle of a batch", "The games already played are still learned from instead of being thrown away."),
        ("S9", "Test games changing the network, or results that cannot be repeated", "Test games switch off learning, and random seeds are fixed (the 500 test games use their own seed)."),
        ("S10", "Running out of memory", "One game is about 24 MB of inputs, so a batch of 4 is easy to hold."),
    ], (0.45, 2.45, 3.6))

    story.append(PageBreak())
    story.append(Paragraph("Pong: Tests", H1))
    story.append(Paragraph("Each test says what we expect to happen, and the result column says what did happen. All 29 tests passed, meaning "
                           "every result matched the expectation. Run them with <font face='Courier'>python test_pong.py</font>. "
                           "The last column says which edge case (S1 to S10) a test covers.", BODY))
    checks_table(story, "A. Scoring and game rules", [
        ("<b>A1.</b> Worked by hand: a point won after 3 pushes and one lost after 2 more should give credits 0.81, 0.9, 1, &minus;0.9, &minus;1.",
         "Those were the credits.", "S1"),
        ("<b>A1b.</b> Without the restart, the earlier pushes should be wrongly affected by the later loss (so the restart really does something).",
         "They were: the first credit fell from 0.81 to 0.15.", "S1"),
        ("<b>A2.</b> Bad settings (batch 0, learning rate 0, discount 1.0 or &minus;0.1) should be rejected.", "All four were rejected.", "S2"),
        ("<b>A3.</b> It should be ALE/Pong-v5 with 6 actions, where button 2 is up (RIGHT) and button 3 is down (LEFT).", "It was.", "S5"),
        ("<b>A4.</b> A full game should end on its own, with rewards only &minus;1, 0 or +1 and a final score between &minus;21 and +21.",
         "A sample game took 944 steps and scored &minus;21.", "&ndash;"),
        ("<b>A5.</b> A game should end when one side reaches 21 points.", "The sample game ended at 0 points won and 21 lost.", "&ndash;"),
    ])
    checks_table(story, "B. What the network sees", [
        ("<b>B1.</b> The first screen of a game should give all zeros.", "The largest value was 0.", "S3"),
        ("<b>B2.</b> Later inputs should show motion: only &minus;1, 0 and +1, with both the +1 (appeared) and &minus;1 (disappeared) kinds present.",
         "True, with about 25 changed pixels per step.", "S3"),
        ("<b>B3.</b> Cleaning a made-up picture should give 80 x 80 values of 0 or 1, with both background colours erased.", "One object pixel was kept.", "&ndash;"),
        ("<b>B4.</b> The game&rsquo;s own picture should not be altered by our use of the cleaning function.", "It was unchanged.", "S4"),
    ])
    checks_table(story, "C. The network", [
        ("<b>C1.</b> Before any training, up and down should be about equally likely.", "The average chance of pushing up was 0.503.", "&ndash;"),
        ("<b>C2.</b> An extreme network (weights multiplied by 10,000) and extreme inputs (millions) should still give finite log-probabilities.", "They did.", "S7"),
        ("<b>C3.</b> One game should fit easily in memory.", "It took 24 MB.", "S10"),
    ])
    checks_table(story, "D. The training loop", [
        ("<b>D1.</b> A game with a NaN reward should make the update skip, leaving the network untouched.", "It was skipped and the weights were identical.", "S6"),
        ("<b>D1b.</b> A NaN inside a game&rsquo;s screen input should make the update skip instead of crashing, leaving the network untouched. "
         "(Added after the same test for the baseline code found that the old code crashed here; it was fixed.)", "It was skipped and the weights were identical.", "S6"),
        ("<b>D2.</b> An empty list of games, or only empty games, should be skipped instead of crashing.", "Both were skipped.", "S6"),
        ("<b>D3.</b> If the limit ends in the middle of a batch (1 game played, batch of 4), that game should still be learned from.", "The network changed.", "S8"),
        ("<b>D4.</b> The same seed should give identical results, and a different seed different ones.", "Two runs with seed 3 matched exactly; seed 4 differed.", "S9"),
        ("<b>D5.</b> Test games should never change the network.", "The weights were identical before and after.", "S9"),
        ("<b>D6.</b> Test games with the same seed should give identical scores.", "They did.", "S9"),
        ("<b>D7.</b> A made-up problem with a known answer (one input where pushing up always pays 1): our learning signal should match the exact gradient of the expected reward.",
         "The direction matched (1.0000) and the size was within 1.8%.", "&ndash;"),
        ("<b>D7b.</b> The previous check should be able to fail: a learner with the sign flipped, or one that ignores the reward, should be caught.",
         "Both were caught (a direction match of &minus;1.00).", "&ndash;"),
        ("<b>D8.</b> In the same made-up problem, training should make the better action much more likely.", "Its chance went from 0.50 to 1.00.", "&ndash;"),
    ])
    checks_table(story, "E. Does it behave as predicted?", [
        ("<b>E1.</b> The saved test result should be exactly 500 games, each scoring between &minus;21 and +21.",
         "500 games, from &minus;7 to 21, a mean of 10.08 and a standard deviation of 4.70.", "&ndash;"),
        ("<b>E2.</b> No learning step should have been skipped in the main run.", "None was, over 6,993 games.", "S6"),
        ("<b>E3.</b> The trained network should be far better than an untrained one (30 fresh games each, statistical test t above 10).",
         "8.6 against &minus;20.2 (t = 28).", "&ndash;"),
        ("<b>E4.</b> Always taking the most likely push should not do much worse than random choice.", "11.2 against 8.6.", "&ndash;"),
        ("<b>E5.</b> The moving average, worked by hand as in the CartPole tests, should come out right.", "All of these matched.", "&ndash;"),
        ("<b>E6.</b> The learning-rate-0.01 run should have frozen because its network became almost always certain (above 99% sure) of its push, "
         "while the main model stays clearly less certain.",
         "99% of the 0.01 model&rsquo;s pushes were that certain (and 89% were up), against 70% for the main model. Its score was &minus;21.", "&ndash;"),
    ])


def pong_graphs_section(story):
    import json
    S = json.load(open(os.path.join(PG_FIG_DIR, "pong_summary.json")))
    main, other, hot = (S["learning rate 0.001, seed 0 (main)"], S["learning rate 0.001, seed 1"], S["learning rate 0.01, seed 0"])
    story.append(PageBreak())
    story.append(Paragraph("(b) Pong-v5: graphs and results", H1))
    figure_from(story, PG_FIG_DIR, "pong_training.png",
                f"Figure 8. Score of every training game (grey) and the moving average of the last 100 games (red), main model "
                f"({main['games']:,} games).", width=4.1 * inch)
    story.append(Paragraph(f"The main model (learning rate 0.001, seed 0) stays near &minus;21, where it loses almost every point, for about the first 2,000 games. "
                           f"Then it takes off, and it levels out at a score of about +8 to +10. Over its last 100 training games it averaged {main['last100']:.1f}.", BODY))
    figure_from(story, PG_FIG_DIR, "pong_histogram.png",
                f"Figure 9. Scores of 500 games played by the trained main model (mean {main['mean']:.2f}, standard deviation {main['std']:.2f}).",
                width=3.6 * inch)
    story.append(Paragraph(f"On the 500 test games the main model scored a mean of {main['mean']:.2f} with a standard deviation of {main['std']:.2f}, "
                           f"with scores from {main['min']:.0f} to {main['max']:.0f}.", BODY))
    figure_from(story, PG_FIG_DIR, "pong_runs.png",
                "Figure 10. All three runs: learning rate 0.001 with two seeds, and the assignment&rsquo;s learning rate 0.01.", width=4.1 * inch)
    story.append(Paragraph(f"<b>Two seeds, similar results.</b> The second seed at the same settings scored a mean of {other['mean']:.2f} on its 500 test games "
                           f"(standard deviation {other['std']:.2f}), against {main['mean']:.2f} for the main model, and both took off at about the same time. "
                           f"Luck still matters for timing: the earlier, shorter laptop runs with these settings (stopped after about 3,300 games) had ended far apart, "
                           f"at &minus;11.55 and +5.65, so two seeds are still not enough to say how much runs differ.", BODY))
    story.append(Paragraph(f"<b>The assignment&rsquo;s learning rate (0.01) did not work.</b> It stayed at &minus;21 for all {hot['games']:,} games "
                           f"(500 test games: mean {hot['mean']:.2f}). We checked why (test E6): its network quickly became almost always certain of its "
                           f"push, mostly up, so it stopped trying anything else and never found out that other pushes score better. "
                           f"This is why we used 0.001, as the assignment allows when training is not stable.", BODY))


def pg_part1_three_months_section(story):
    block = []
    story_real, story = story, block
    story.append(Paragraph("Policy gradient Part 1: With 3 more months", H1))
    story.append(Paragraph("<b>Be more sure of the results.</b> Our runs depend a lot on luck: at the same settings, shorter Pong runs of two seeds ended far apart, and "
                           "CartPole seeds ranged from 155 to 500. With more time we would run 20 or more seeds for each setting, report averages with "
                           "confidence ranges, and test whether differences are real or just luck. We would also train Pong for much longer, with a GPU "
                           "that does not sleep, to see where it levels off.", BODY))
    story.append(Paragraph("<b>Test our own choices.</b> We chose to restart the credit at each Pong point, to show the network frame differences, and to "
                           "learn from 4 games at a time. We would switch each choice off in turn to show that it helps. We would also search "
                           "properly over learning rate, network size and batch size, and look at why the assignment&rsquo;s learning rate of 0.01 "
                           "froze on Pong (for example by capping the size of each step, or rewarding some randomness).", BODY))
    story.append(Paragraph("<b>Check it a different way.</b> We would try a convolutional network on the Pong pictures, test the trained player against "
                           "a faster opponent and from different starting points to see whether it only learned one narrow trick, and watch recordings of its "
                           "games to see how it plays.", BODY))
    story_real.append(KeepTogether(block))


def pg_part2_three_months_section(story):
    block = []
    story_real, story = story, block
    story.append(Paragraph("Policy gradient Part 2 (baselines): With 3 more months", H1))
    story.append(Paragraph("<b>Prove the baseline helps.</b> We would repeat the comparison with 20 or more seeds, on equal numbers of games and on equal "
                           "training time, and use a proper statistical test for &ldquo;baseline against no baseline&rdquo;. We would also plot results "
                           "against game steps, not just games, because games get longer as the player improves.", BODY))
    story.append(Paragraph("<b>Measure the noise directly.</b> We checked on a made-up problem that a baseline cuts the noise of each learning step without "
                           "changing where learning points. With more time we would measure the same thing during real training, by recording how much "
                           "the learning step varies with and without a baseline, and check there is no hidden bias when the baseline is learned "
                           "from the same games.", BODY))
    story.append(Paragraph("<b>Try other baselines.</b> We would compare other references: an average over the batch, a learned baseline with a different size or "
                           "learning rate, rescaling the credits, and a baseline that shares its first layers with the policy. We would also try a "
                           "convolutional network for the Pong baseline.", BODY))
    story_real.append(KeepTogether(block))


def baseline_intro_section(story):
    story.append(PageBreak())
    story.append(Paragraph("Question 3: Policy Gradient, Part 2 (baselines)", TITLE))
    story.append(Paragraph("<b>What a baseline is.</b> In Part 1, every push was credited with all the reward that followed it. That is noisy: "
                           "a push can look good just because the whole stretch around it went well, and in CartPole every push is followed by "
                           "plenty of reward early in a game, even the bad ones. A baseline is a reference number we subtract first, so that "
                           "<i>credit = reward that followed &minus; reference</i>. A push now counts as good only if things went better than the "
                           "reference, and as bad if they went worse.", BODY))
    story.append(Paragraph("<b>Why this helps.</b> The reference must not depend on the push being judged. Then, averaged over many games, "
                           "learning still points the same way as before, but each learning step is much less noisy. Less noise means steadier "
                           "learning, and usually faster learning too.", BODY))
    story.append(Paragraph("<b>The two baselines we use.</b> A <i>constant</i> baseline is one number for every situation: a running average of the "
                           "rewards from earlier games. A <i>learned</i> baseline is a second small network that guesses, for each situation, how much "
                           "reward to expect from there on, so it can tell an easy moment from a hard one. CartPole uses both; Pong uses the learned one.", BODY))
    story.append(Paragraph("Assumptions", H1))
    story.append(Paragraph("These are the choices we made for the baseline experiments on CartPole-v1:", BODY))
    assumptions = [
        "<b>Only the baseline changes.</b> The policy network, discount (0.95), learning rate (0.01), number of games (1500), "
        "one update per game, and the random seeds are exactly as in Part 1, so any difference comes from the baseline. "
        "&ldquo;No baseline&rdquo; in this part is exactly Part 1 (a test checks that the training curves are identical).",
        "<b>The constant baseline</b> is a running average of the rewards from earlier games (about the last 20). Before the first game it is 0. "
        "We use earlier games only, because an average that included the current game would depend slightly on the pushes being judged.",
        "<b>The learned baseline</b> is a second network of the same size as the policy (4 inputs, one hidden layer of 128 units, 1 output). "
        "It is trained after each game to guess the rewards that were actually earned (squared error), with its own Adam optimiser at the same learning rate. "
        "Its guesses are only used as a reference: they carry no learning information into the policy.",
        "<b>The same credit as before otherwise:</b> each push is credited with the reward from that step onwards, discounted by 0.95, "
        "minus the reference for that step. Credits are not rescaled.",
        "<b>Ten seeds for each method.</b> A first run with three seeds was too noisy to compare the methods (individual runs at this learning "
        "rate can take off or collapse), so we used ten seeds each. The main model (learned baseline, seed 0) was chosen in advance; "
        "the other seeds show how much luck matters.",
        "<b>Final test:</b> as in Part 1, 500 new games with random choices and a different random seed.",
    ]
    for a in assumptions:
        story.append(Paragraph(a, ParagraphStyle("li5", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=4), bulletText="•"))


def cartpole_baseline_edge_cases_section(story):
    story.append(PageBreak())
    story.append(Paragraph("CartPole baselines: Edge cases", H1))
    story.append(Paragraph("The unusual situations we considered for the baseline experiments, and how the code handles each one (U1 to U9).", BODY))
    table4(story, ("#", "Unusual situation", "How the code handles it"), [
        ("U1", "A baseline that depends on the push it judges (this would bend the direction of learning)",
         "The constant baseline uses earlier games only. The learned baseline is used without learning information, so it cannot be pulled around by the policy&rsquo;s own step."),
        ("U2", "A NaN or infinite number reaching the baseline network (for example in a game&rsquo;s states)", "The update is skipped, and neither network is changed. The baseline&rsquo;s own training step is skipped too if its error is not a usable number."),
        ("U3", "A NaN or infinite reward", "The whole update is skipped, and neither the policy nor the baseline network is changed."),
        ("U4", "Bad settings, such as an unknown baseline name, 0 games, or a discount of 1 or more", "Rejected at the start with a clear message."),
        ("U5", "An empty game, or a game of only one step", "An empty game is skipped. A one-step game still gives a valid update for both baselines."),
        ("U6", "The new code accidentally changing Part 1", "&ldquo;No baseline&rdquo; subtracts a reference of exactly 0, and a test checks that it reproduces the Part 1 training curve game for game."),
        ("U7", "The first games, before the constant baseline has anything to average", "It starts at 0, so the first update is the same as in Part 1."),
        ("U8", "Results that cannot be repeated", "Random seeds are fixed, and the 500 test games use a different seed from training."),
        ("U9", "One lucky or unlucky run deciding the comparison", "Ten seeds for each method, and every seed is shown."),
    ], (0.45, 2.45, 3.6))

    story.append(PageBreak())
    story.append(Paragraph("CartPole baselines: Tests", H1))
    story.append(Paragraph("Each test says what we expect to happen, and the result column says what did happen. "
                           "Run them with <font face='Courier'>python test_cartpole_baselines.py</font>. "
                           "The last column says which edge case (U1 to U9) a test covers.", BODY))
    checks_table(story, "A. The reference and the credits", [
        ("<b>A1.</b> Worked by hand: rewards 3, 2, 1 against a reference of 2 should give credits +1, 0, &minus;1, so the number we minimise is &minus;1.5.",
         "It was &minus;1.50.", "&ndash;"),
        ("<b>A2.</b> &ldquo;No baseline&rdquo; should be exactly Part 1: the same training curve, game for game.", "60 games were compared and all were identical.", "U6"),
        ("<b>A3.</b> In all three settings, the learning step should be exactly the hand-written one (credit = reward &minus; reference).", "It was.", "U1"),
        ("<b>A3b.</b> Before the first game the constant baseline is 0, so the first update should be identical to &ldquo;no baseline&rdquo;.", "It was identical.", "U7"),
        ("<b>A4.</b> The learned baseline should be only a reference for the credits, with no learning information carried into the policy.",
         "Its first use carried none; its second use (its own training) did.", "U1"),
        ("<b>A5.</b> Bad settings (0 games, learning rate 0, discount 1.0, an unknown baseline name) should be rejected.", "All were rejected.", "U4"),
    ])
    checks_table(story, "B. A baseline should not change where learning points, only how noisy it is", [
        ("<b>B1.</b> Made-up problem with a known answer (one situation where pushing 0 pays 10 and pushing 1 pays 12): with no baseline, a constant baseline and a "
         "learned baseline, our learning signal should match the exact gradient of the expected reward.",
         "All three matched: direction match 1.0000 in each, with size errors of 0.092 (none) and 0.001 (constant and learned).", "U1"),
        ("<b>B1b.</b> The previous check should be able to fail: a baseline that peeks at the push&rsquo;s own reward, or subtracts twice as much, should be caught.",
         "Both were caught (direction match 0.00 and &minus;1.00).", "U1"),
        ("<b>B2.</b> A baseline should make the learning signal much less noisy (repeated small batches of 20 pushes).",
         "The typical spread was 4.95 with no baseline and 1.42 with either baseline.", "&ndash;"),
        ("<b>B3.</b> The learned baseline network should learn to expect about the right reward in the made-up problem (about 11).", "Its guess was 11.00.", "&ndash;"),
    ])
    checks_table(story, "C. The training loop", [
        ("<b>C1.</b> A NaN reward should make the update skip, leaving both networks untouched (constant and learned baselines).",
         "It was skipped and all weights were identical.", "U3"),
        ("<b>C1b.</b> A NaN inside a game&rsquo;s states (so the learned baseline outputs NaN) should make the update skip, leaving both networks untouched.",
         "It was skipped and all weights were identical.", "U2"),
        ("<b>C2.</b> An empty game should be skipped instead of crashing (all three settings).", "It was skipped.", "U5"),
        ("<b>C3.</b> A one-step game should still give a valid update (constant and learned baselines).", "It did, with all numbers finite.", "U5"),
        ("<b>C4.</b> The same seed should give an identical training curve, and a different seed a different one (constant and learned baselines).",
         "Same seed identical, different seed different.", "U8"),
    ])
    checks_table(story, "D. Does it behave as predicted? (ten seeds per method)", [
        ("<b>D1.</b> The 500 test scores of every run should be valid (between 1 and 500).", "All 30 runs were, from 8 to 500.", "&ndash;"),
        ("<b>D2.</b> The learned baseline should do better than no baseline, and the difference should be unlikely to be luck (two statistical tests, both p below 0.05).",
         "330.9 against 154.9, with p = 0.018 and 0.008.", "U9"),
        ("<b>D3.</b> The same should hold for the constant baseline.", "482.1 against 154.9, with p below 0.001 in both tests.", "U9"),
        ("<b>D4.</b> The saved main model should reproduce exactly the 500 test scores behind the graphs.", "All 500 scores were identical (with 1 thread).", "U8"),
        ("<b>D5.</b> Retraining the main model from the same seed should repeat the start of the saved training curve.",
         "The first 200 training games were identical (with 1 thread).", "U8"),
    ])


BASE_FIG_DIR = os.path.join(HERE, "..", "q4_baselines", "figures")


def cartpole_baseline_graphs_section(story):
    import json
    from scipy import stats
    S = json.load(open(os.path.join(BASE_FIG_DIR, "cartpole_baseline_summary.json")))
    seeds = sorted({int(k.split("seed")[1]) for k in S})
    per = lambda b, key: np.array([S[f"{b}_seed{s}"][key] for s in seeds])
    names = {"none": "No baseline", "constant": "Constant baseline", "learned": "Learned baseline"}
    p = lambda a, b: stats.ttest_ind(per(a, "mean"), per(b, "mean"), equal_var=False).pvalue
    main = S["learned_seed0"]

    story.append(PageBreak())
    story.append(Paragraph("(a) CartPole-v1 with baselines: graphs and results", H1))
    story.append(Paragraph("The main model is the learned baseline with seed 0, chosen in advance. Figures 11 and 12 show it; "
                           "Figures 13 and 14 compare all three methods over ten seeds each.", BODY))
    figure_from(story, BASE_FIG_DIR, "cartpole_baseline_training.png",
                "Figure 11. Reward of every training game (grey) and the moving average of the last 100 games (red), main model.", width=4.1 * inch)
    figure_from(story, BASE_FIG_DIR, "cartpole_baseline_histogram.png",
                f"Figure 12. Scores of 500 games played by the trained main model (mean {main['mean']:.1f}, standard deviation {main['std']:.1f}).", width=3.6 * inch)
    story.append(Paragraph(f"The main model is not a good advertisement for the method: it scored a mean of {main['mean']:.1f} on the 500 test games, "
                           f"which is below the average of its method. We report it anyway, because we chose it before seeing any results. "
                           f"The comparison over ten seeds is the better guide.", BODY))
    figure_from(story, BASE_FIG_DIR, "cartpole_baseline_comparison.png",
                "Figure 13. The three methods during training. Faint lines are single seeds, bold lines are the average over ten seeds.", width=4.3 * inch)
    figure_from(story, BASE_FIG_DIR, "cartpole_baseline_seeds.png",
                "Figure 14. The test score of every seed (500 test games each); the black line is the average of the ten seeds.", width=3.6 * inch)
    rows = []
    for b in ("none", "constant", "learned"):
        m = per(b, "mean")
        rows.append((names[b], f"{m.mean():.1f}", f"{m.min():.0f} to {m.max():.0f}", f"{int((m >= 499.5).sum())} of {len(m)}"))
    table4(story, ("Method", f"Average test score ({len(seeds)} seeds)", "Worst and best seed", "Seeds at the 500 limit"), rows, (1.9, 1.8, 1.6, 1.2))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Both baselines help.</b> The constant baseline averaged {per('constant', 'mean').mean():.0f} against "
                           f"{per('none', 'mean').mean():.0f} with no baseline (p &lt; 0.001), and the learned baseline averaged {per('learned', 'mean').mean():.0f} "
                           f"(p = {p('learned', 'none'):.3f}). A second statistical test, which makes fewer assumptions, agrees. "
                           f"Only {int((per('none', 'mean') >= 499.5).sum())} of {len(seeds)} seeds without a baseline reached the 500 limit, against "
                           f"{int((per('constant', 'mean') >= 499.5).sum())} with the constant baseline and {int((per('learned', 'mean') >= 499.5).sum())} "
                           f"with the learned one. Training is still bumpy with a baseline: the curves in Figure 13 rise and fall.", BODY))
    story.append(Paragraph(f"<b>A surprise.</b> We expected the learned baseline to beat the constant one, because it can tell easy moments from hard ones. "
                           f"It did not: the constant baseline was better (p = {p('constant', 'learned'):.3f}). We did not investigate why. "
                           f"One possible reason is that the learned network is trained with a single noisy step after each game at a high learning rate, "
                           f"so its guesses lag behind. This is a guess, not something we tested, and with more tuning the order could change.", BODY))
    story.append(Paragraph("<b>Luck matters a lot, and even tiny differences matter.</b> A first version of this experiment used three seeds and made the learned "
                           "baseline look worse than no baseline. With ten seeds the picture reversed, so three seeds were not enough. "
                           "We also found that running the same seed with a different number of computer threads can change a run completely, "
                           "because tiny rounding differences grow over 1,500 games of training. All results here were made with 1 thread per run, "
                           "and the tests use the same setting.", BODY))


def pong_baseline_edge_cases_section(story):
    story.append(PageBreak())
    story.append(Paragraph("Pong baseline: Edge cases", H1))
    story.append(Paragraph("The unusual situations we considered for the Pong baseline experiment, and how the code handles each one (V1 to V9). "
                           "Pong uses only the learned baseline: a second network that sees the same screen-difference as the policy.", BODY))
    table4(story, ("#", "Unusual situation", "How the code handles it"), [
        ("V1", "A baseline that depends on the push it judges (this would bend the direction of learning)",
         "The baseline network is used without learning information, so the policy&rsquo;s step cannot pull it around. It is trained separately."),
        ("V2", "A NaN or infinite number in a game&rsquo;s screen input or rewards", "The update is skipped before either network is used, and neither is changed. (Writing this test found that the old code crashed instead of skipping; it was fixed, in Part 1 as well.)"),
        ("V3", "The baseline network&rsquo;s own training producing an unusable number", "That training step is skipped, and the policy is not affected. (This protection is in the code but is not tested on its own, because an unusable input is already caught earlier; see V2.)"),
        ("V4", "Bad settings: an unknown baseline name (the constant baseline is CartPole only), batch below 1, learning rate of 0 or below, discount of 1 or more", "Rejected at the start with a clear message."),
        ("V5", "An empty game, or a game of only one step", "Empty games are skipped. A one-step game still gives a valid update."),
        ("V6", "The new code accidentally changing Part 1", "&ldquo;No baseline&rdquo; subtracts a reference of exactly 0, and a test checks that one learning step gives identical weights to the Part 1 code."),
        ("V7", "Credit leaking across points", "As in Part 1, the credit restarts at each point, and the baseline is trained to guess the same restarted credit."),
        ("V8", "Running out of time or games in the middle of a batch", "The games already played are still learned from."),
        ("V9", "Results that cannot be repeated", "Random seeds are fixed, and the 500 test games use their own seed."),
    ], (0.45, 2.45, 3.6))

    story.append(PageBreak())
    story.append(Paragraph("Pong baseline: Tests", H1))
    story.append(Paragraph("Each test says what we expect to happen, and the result column says what did happen. "
                           "Run them with <font face='Courier'>python test_pong_baseline.py</font>. "
                           "The last column says which edge case (V1 to V9) a test covers.", BODY))
    checks_table(story, "A. The reference and the credits", [
        ("<b>A1.</b> The learning step should be exactly the hand-written one: credit = (reward that followed, restarting at each point) &minus; reference, averaged over the games, with and without the learned baseline.",
         "It was.", "V1, V7"),
        ("<b>A2.</b> &ldquo;No baseline&rdquo; should be exactly Part 1: one learning step should give identical weights.", "The weights were identical.", "V6"),
        ("<b>A3.</b> The learned baseline should be only a reference for the credits, with no learning information carried into the policy.",
         "Its first use carried none; its second use (its own training) did.", "V1"),
        ("<b>A4.</b> Bad settings (batch 0, learning rate 0, discount 1.0, an unknown baseline name) should be rejected.", "All four were rejected.", "V4"),
    ])
    checks_table(story, "B. A baseline should not change where learning points, only how noisy it is", [
        ("<b>B1.</b> The learned baseline network should learn to expect about the right reward in a made-up problem (pushing down pays 10, pushing up pays 12, so about 11).",
         "Its guess was 11.11.", "&ndash;"),
        ("<b>B2.</b> Made-up problem with a known answer: with no baseline and with a learned baseline, our learning signal should match the exact gradient of the expected reward.",
         "Both matched: direction match 1.0000 in each, with size errors of 0.066 (none) and 0.001 (learned).", "V1"),
        ("<b>B2b.</b> The previous check should be able to fail: a baseline that peeks at the push&rsquo;s own reward, or subtracts twice as much, should be caught.",
         "Both were caught (direction match 0.00 and &minus;1.00).", "V1"),
        ("<b>B3.</b> A baseline should make the learning signal much less noisy (repeated small batches of 20 pushes).",
         "The typical spread was 8.44 with no baseline and 3.18 with the learned baseline.", "&ndash;"),
    ])
    checks_table(story, "C. The training loop", [
        ("<b>C1.</b> A NaN reward should make the update skip, leaving both networks untouched.", "It was skipped and all weights were identical.", "V2"),
        ("<b>C1b.</b> A NaN inside a game&rsquo;s screen input should make the update skip instead of crashing, leaving both networks untouched.",
         "It was skipped and all weights were identical (after the fix described under V2).", "V2"),
        ("<b>C2.</b> An empty list of games, or only empty games, should be skipped instead of crashing (both settings).", "They were skipped.", "V5"),
        ("<b>C3.</b> A one-step game should still give a valid update.", "It did, with all numbers finite.", "V5"),
        ("<b>C4.</b> The same seed should give an identical training curve, and a different seed a different one (real games).",
         "Same seed identical, different seed different.", "V9"),
        ("<b>C5.</b> If the game limit ends in the middle of a batch (1 game played, batch of 4), that game should still be learned from.", "The network changed.", "V8"),
    ])


SECTIONS = [summary_section, shuttle_section, edge_cases_section, three_months_section, policy_gradient_section, cartpole_edge_cases_section, pong_section, pong_graphs_section, pong_edge_cases_section, pg_part1_three_months_section, baseline_intro_section, cartpole_baseline_graphs_section, cartpole_baseline_edge_cases_section, pong_baseline_edge_cases_section]

story = []
for add in SECTIONS:
    add(story)
SimpleDocTemplate(os.path.join(HERE, "report.pdf"), pagesize=letter, leftMargin=inch, rightMargin=inch,
                  topMargin=0.9 * inch, bottomMargin=0.9 * inch, title="Advanced ML Assignment 1").build(story)
print("wrote report.pdf")
